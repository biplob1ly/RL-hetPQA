"""
train model
Usage:
    dpr_trainer.py  --path_cfg_exp=<path> [--path_data=<path>] [--path_model=<path>] [--path_output=<path>] [--version=<val>] [--dpr_ckpt=<filename>]
    dpr_trainer.py -h | --help

Options:
    -h --help                   show this screen help
    --path_cfg_exp=<path>       experiment config path
    --path_data=<path>          data path
    --path_model=<path>         model path
    --path_output=<path>        output path
    --path_train_data=<path>    train data path
    --path_val_data=<path>      validation data path
    --path_test_data=<path>     Test data path
    --version=<val>             version
    --dpr_ckpt=<filename>       DPR checkpoint file name
"""
from docopt import docopt
import os
import shutil
from datetime import datetime
import torch
import logging
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from configs.dpr_config.config import get_cfg_defaults

from dpr.models.biencoder import BiEncoderNllLoss, Interaction
from dpr.utils.model_utils import (
    get_model_components, get_optimizer_components, get_model_obj,
    setup_for_distributed_mode, get_model_file, load_states_from_checkpoint,
    CheckpointState, set_model_cfg_from_state, get_model_params_state
)
from dpr.options import setup_cfg_gpu, set_seed
from dpr.utils.data_utils import RetDataset, RetCollator
from dpr_utils import (
    save_ranking_results, save_combined_results, save_eval_metrics, compute_metrics,
    DPRValResult, format_dpr_run, get_ranked_ctxs
)

logging.basicConfig(
    filename='dpr_logs.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieverTrainer:
    def __init__(self, cfg, model_file=None):
        self.cfg = cfg
        self.shard_id = cfg.LOCAL_RANK if cfg.LOCAL_RANK != -1 else 0
        self.distributed_factor = cfg.DISTRIBUTED_WORLD_SIZE or 1

        logger.info("***** Initializing model components *****")
        # if model file is specified, encoder parameters from saved state should be used for initialization
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            # Not sure if needed. It overwrites cfg params from checkpoint states
            set_model_cfg_from_state(saved_state.model_params, cfg)

        tokenizer, biencoder = get_model_components(cfg)
        optimizer, scheduler = get_optimizer_components(cfg, biencoder)
        model, optimizer = setup_for_distributed_mode(biencoder, optimizer, cfg.DEVICE, cfg.N_GPU,
                                                      cfg.LOCAL_RANK,
                                                      cfg.FP16,
                                                      cfg.FP16_OPT_LEVEL)
        self.tokenizer = tokenizer
        self.biencoder = biencoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_step = 0
        self.scheduler_state = None
        self.validations = []
        self.saved_cps = {}
        self.best_cp_name = None
        self.train_dataset = None
        self.val_dataset = None
        self.collator = RetCollator(
            tokenizer=tokenizer,
            question_max_len=cfg.DPR.MODEL.QUESTION_MAX_LENGTH,
            ctx_max_len=cfg.DPR.MODEL.CONTEXT_MAX_LENGTH
        )
        self.loss_function = BiEncoderNllLoss(
            level=cfg.DPR.SOLVER.LEVEL,
            broadcast=cfg.DPR.SOLVER.BROADCAST,
            func=cfg.DPR.SOLVER.FUNC,
            temperature=cfg.DPR.SOLVER.TEMPERATURE
        )
        if saved_state:
            strict = not cfg.DPR.MODEL.PROJECTION_DIM
            self._load_saved_state(saved_state, strict=strict)

    def get_representations(
            self,
            q_input_ids, q_attention_mask, q_token_type_ids,
            ctx_input_ids, ctx_attention_mask, ctx_token_type_ids
    ):
        cfg = self.cfg
        device = cfg.DEVICE
        model_output = self.biencoder(
            q_input_ids=q_input_ids.to(device) if q_input_ids is not None else None,
            q_attention_mask=q_attention_mask.to(device) if q_attention_mask is not None else None,
            q_token_type_ids=q_token_type_ids.to(device) if q_token_type_ids is not None else None,
            ctx_input_ids=ctx_input_ids.to(device),
            ctx_attention_mask=ctx_attention_mask.to(device),
            ctx_token_type_ids=ctx_token_type_ids.to(device)
        )
        if cfg.DPR.SOLVER.LEVEL == "pooled":
            # q_repr: 1 x S x D
            q_reprs = model_output.q_pooled.detach().cpu() if q_input_ids is not None else None
            # ctx_repr: sub_batch_size x S x D
            ctx_reprs = model_output.ctx_pooled.detach().cpu()
        else:
            q_reprs = model_output.q_seq.detach().cpu() if q_input_ids is not None else None
            ctx_reprs = model_output.ctx_seq.detach().cpu()
        return q_reprs, ctx_reprs

    def evaluate(self, eval_dataset: RetDataset):
        logger.info('Evaluating ranker ...')
        self.biencoder.eval()
        cfg = self.cfg
        eval_sampler = SequentialSampler(eval_dataset)
        eval_data_loader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=cfg.DPR.SOLVER.TEST_BATCH_SIZE,
            drop_last=False,
            collate_fn=self.collator
        )
        sub_batch_size = cfg.DPR.SOLVER.TEST_CTX_BSZ
        # embed_size = self.biencoder.ctx_model.get_out_size()
        interaction = Interaction(
            level=cfg.DPR.SOLVER.LEVEL,
            broadcast=cfg.DPR.SOLVER.BROADCAST,
            func=cfg.DPR.SOLVER.FUNC
        )
        result_data = []
        with torch.no_grad():
            for iteration, batch in enumerate(eval_data_loader):
                # qids, cids_per_qid, srcs_per_qid, pos_cids_per_qid,
                # q_input_ids, q_attention_mask, q_token_type_ids,
                # ctx_input_ids, ctx_attention_mask, ctx_token_type_ids
                # Dim: (Q*C) x S
                ctx_input_ids = batch.ctx_input_ids
                ctx_attention_mask = batch.ctx_attention_mask
                ctx_token_type_ids = batch.ctx_token_type_ids
                bsz = ctx_input_ids.size(0)
                q_reprs = None
                scores = []
                for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):
                    # Dim: 1 x S
                    q_input_ids, q_attention_mask, q_token_type_ids = (
                        (batch.q_input_ids, batch.q_attention_mask, batch.q_token_type_ids) if j == 0 else (None, None, None)
                    )
                    # Dim: sub_batch_size x S
                    sub_ctx_input_ids = ctx_input_ids[batch_start:batch_start + sub_batch_size]
                    sub_ctx_attention_mask = ctx_attention_mask[batch_start:batch_start + sub_batch_size]
                    sub_ctx_token_type_ids = ctx_token_type_ids[batch_start:batch_start + sub_batch_size]

                    sub_q_reprs, sub_ctx_reprs = self.get_representations(
                        q_input_ids, q_attention_mask, q_token_type_ids,
                        sub_ctx_input_ids, sub_ctx_attention_mask, sub_ctx_token_type_ids
                    )
                    if sub_q_reprs is not None:
                        # q_dense: 1 x S x D
                        q_reprs = sub_q_reprs
                    scores.extend(interaction.compute_score(q_reprs, sub_ctx_reprs).flatten().tolist())
                    # if q_dense is not None:
                    #     q_embeds.extend(q_dense.cpu().split(1, dim=0))
                    # ctx_embeds.extend(ctx_dense.cpu().split(1, dim=0))
                # q_embeds = torch.cat(q_embeds, dim=0)
                # ctx_embeds = torch.cat(ctx_embeds, dim=0)
                # assert len(ctx_embeds) == len(ctx_ids)

                # indexer = DenseFlatIndexer(embed_size)
                # indexer.index_data(ctx_embeds.numpy(), np.array([int(ctx_id) for ctx_id in ctx_ids]))
                # ctx_scores_arr, ctx_ids_arr = indexer.search_knn(q_embeds.numpy(), len(ctx_ids))
                # ctx_scores_list, ctx_ids_list = ctx_scores_arr[0].tolist(), ctx_ids_arr[0].tolist()
                # pred_ctx_ids = [str(ctx_id) for ctx_id in ctx_ids_list]

                # scores: C
                ctx_scores_list, pred_ctx_ids, pred_ctx_srcs = get_ranked_ctxs(
                    scores,
                    batch.cids_per_qid[0],
                    batch.srcs_per_qid[0]
                )
                result_data.append(
                    {
                        'qid': batch.qids[0],
                        'pred_ctx_sources': pred_ctx_srcs,
                        'scores': ctx_scores_list,
                        'pred_ctx_ids': pred_ctx_ids,
                        'actual_ctx_ids': batch.pos_cids_per_qid[0]
                    }
                )
        return result_data

    def _save_checkpoint(self, step: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(cfg.DPR.MODEL.MODEL_PATH,
                          cfg.DPR.MODEL.CHECKPOINT_FILE_NAME + '.' + str(step))

        meta_params = get_model_params_state(cfg)
        state = CheckpointState(
            model_to_save.state_dict(),
            meta_params,
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
            step
        )
        torch.save(state._asdict(), cp)
        logger.info('Saved checkpoint at %s', cp)
        return cp

    def validate_and_save(self, cur_step: int, val_dataset: RetDataset):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.LOCAL_RANK in [-1, 0]

        cur_val_id = len(self.validations)
        if cfg.DPR.DATA.VAL_DATA_PATH:
            # validation_loss = self.validate_nll()
            result_list = self.evaluate(val_dataset)
            val_metrics = ["map", "r-precision", "mrr", "ndcg", "hit_rate@5", "precision@1"]
            metrics_dt = compute_metrics(result_list, val_metrics)['all']
            metrics_score = [metrics_dt[metric] for metric in val_metrics]
            ret_eval = DPRValResult(cur_val_id, cur_step, val_metrics, metrics_score)
            self.validations.append(ret_eval)
            fmt_header, fmt_value = format_dpr_run(ret_eval)
            logger.info(fmt_header)
            logger.info(fmt_value)
            if cur_val_id == 0:
                print(fmt_header)
            print(fmt_value)

        if save_cp:
            best_ret_eval = max(self.validations, key=lambda x: x.scores)
            if len(self.saved_cps) < cfg.DPR.SOLVER.CP_SAVE_LIMIT:
                cp_path = self._save_checkpoint(cur_step)
                self.saved_cps[cur_val_id] = cp_path
                if best_ret_eval.val_id == cur_val_id:
                    self.best_cp_name = cp_path
                    logger.info('New Best validation checkpoint %s', cp_path)
            else:
                sorted_runs = sorted(self.validations, key=lambda x: x.scores, reverse=True)
                for ret_eval in sorted_runs[cfg.DPR.SOLVER.CP_SAVE_LIMIT:]:
                    if ret_eval.val_id in self.saved_cps:
                        os.remove(self.saved_cps[ret_eval.val_id])
                        del self.saved_cps[ret_eval.val_id]
                        cp_path = self._save_checkpoint(cur_step)
                        self.saved_cps[cur_val_id] = cp_path
                        if best_ret_eval.val_id == cur_val_id:
                            self.best_cp_name = cp_path
                            logger.info('New Best validation checkpoint %s', cp_path)
                        break

    def calculate_loss(self, batch):
        device = self.cfg.DEVICE
        model_output = self.biencoder(
            q_input_ids=batch.q_input_ids.to(device),
            q_attention_mask=batch.q_attention_mask.to(device),
            q_token_type_ids=batch.q_token_type_ids.to(device),
            ctx_input_ids=batch.ctx_input_ids.to(device),
            ctx_attention_mask=batch.ctx_attention_mask.to(device),
            ctx_token_type_ids=batch.ctx_token_type_ids.to(device)
        )
        cur_loss = self.loss_function.compute_loss(
            model_output=model_output,
            cids_per_qid=batch.cids_per_qid,
            pos_cids_per_qid=batch.pos_cids_per_qid
        )
        return cur_loss

    def train(self, train_dataset, val_dataset=None):
        self.biencoder.train()
        cfg = self.cfg
        train_sampler = RandomSampler(train_dataset)
        train_data_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=cfg.DPR.SOLVER.TRAIN_BATCH_SIZE,
            drop_last=True,
            collate_fn=self.collator
        )

        logger.info("Total updates=%d", cfg.DPR.SOLVER.TOTAL_TRAIN_STEPS)
        logger.info("Eval step = %d", cfg.DPR.SOLVER.NUM_STEP_PER_EVAL)
        logger.info("***** Training *****")
        cur_step = self.start_step
        rolling_loss = 0
        epoch = 0
        last_saved_step = -1
        while cur_step < cfg.DPR.SOLVER.TOTAL_TRAIN_STEPS:
            epoch += 1
            logger.info("***** Epoch %d *****", epoch)
            for iteration, batch in enumerate(train_data_loader):
                cur_loss = self.calculate_loss(batch)
                if self.cfg.DPR.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS > 1:
                    cur_loss = cur_loss / self.cfg.DPR.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS
                rolling_loss += cur_loss.item()
                cur_loss.backward()
                if (iteration + 1) % self.cfg.DPR.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.biencoder.parameters(), cfg.DPR.SOLVER.OPTIMIZER.MAX_GRAD_NORM)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.biencoder.zero_grad()
                    cur_step += 1

                if cur_step % cfg.DPR.SOLVER.NUM_STEP_PER_EVAL == 0 and last_saved_step != cur_step:
                    logger.info(
                        "Rank=%d, step: %d/%d, avg train loss: %f, lr: %f",
                        cfg.LOCAL_RANK,
                        cur_step,
                        cfg.DPR.SOLVER.TOTAL_TRAIN_STEPS,
                        rolling_loss/cfg.DPR.SOLVER.NUM_STEP_PER_EVAL,
                        self.scheduler.get_last_lr()[0]
                    )
                    self.validate_and_save(cur_step, val_dataset)
                    self.biencoder.train()
                    rolling_loss = 0
                    last_saved_step = cur_step
                if cur_step >= cfg.DPR.SOLVER.TOTAL_TRAIN_STEPS:
                    break

        logger.info(
            "Rank=%d, step: %d/%d, avg train loss: %f, lr: %f",
            cfg.LOCAL_RANK,
            cur_step,
            cfg.DPR.SOLVER.TOTAL_TRAIN_STEPS,
            rolling_loss / cfg.DPR.SOLVER.NUM_STEP_PER_EVAL,
            self.scheduler.get_last_lr()[0]
        )
        self.validate_and_save(cur_step, val_dataset)
        logger.info("********** Training Completed **********")
        if cfg.LOCAL_RANK in [-1, 0]:
            for idx, dpr_val_result in enumerate(self.validations):
                fmt_header, fmt_value = format_dpr_run(dpr_val_result)
                if idx == 0:
                    logger.info(fmt_header)
                logger.info(fmt_value)
            logger.info("Training finished. Best validation checkpoint %s", self.best_cp_name)
        return self.best_cp_name

    def _load_saved_state(self, saved_state: CheckpointState, strict=True):
        if self.cfg.DPR.SOLVER.RESET_CHECKPOINT_STEP:
            self.step = 0
            logger.info('Resetting checkpoint step=%s', self.step)
        else:
            self.step = saved_state.step
            logger.info('Loading checkpoint step=%s', self.step)

        model_to_load = get_model_obj(self.biencoder)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict, strict=strict)

        if not self.cfg.DPR.SOLVER.OPTIMIZER.RESET:
            if saved_state.optimizer_dict:
                logger.info('Loading saved optimizer state ...')
                self.optimizer.load_state_dict(saved_state.optimizer_dict)
            if saved_state.scheduler_dict:
                logger.info("Loading scheduler state %s", saved_state.scheduler_dict)
                self.scheduler.load_state_dict(saved_state.scheduler_dict)


def run(cfg):
    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg.SEED)

    if cfg.DPR.DO_TRAIN:
        config.DPR.DATA.TRAIN_DATA_PATH = os.path.join(data_path, 'separated', 'norm_train.json')
        config.DPR.DATA.VAL_DATA_PATH = os.path.join(data_path, 'mixed', 'norm_dev.json')
        model_file = get_model_file(cfg, cfg.DPR.MODEL.CHECKPOINT_FILE_NAME)
        retriever_trainer = RetrieverTrainer(cfg, model_file=model_file)
        train_dataset = RetDataset(
            file=cfg.DPR.DATA.TRAIN_DATA_PATH,
            num_pos_ctx=cfg.DPR.DATA.NUM_POSITIVE_CONTEXTS,
            num_total_ctx=cfg.DPR.DATA.NUM_TOTAL_CONTEXTS,
            normalize=cfg.DPR.DATA.NORMALIZE,
            flatten_attr=cfg.DPR.DATA.FLATTEN_ATTRIBUTE,
            is_train=True
        )
        val_dataset = RetDataset(
            file=cfg.DPR.DATA.VAL_DATA_PATH,
            num_pos_ctx=None,
            num_total_ctx=None,
            normalize=cfg.DPR.DATA.NORMALIZE,
            flatten_attr=cfg.DPR.DATA.FLATTEN_ATTRIBUTE,
            is_train=False
        )
        best_cp_path = retriever_trainer.train(train_dataset, val_dataset=val_dataset)
        cfg.dump(stream=open(os.path.join(cfg.DPR.MODEL.MODEL_PATH, f'config_{cfg.EXP}.yaml'), 'w'))
        cfg.DPR.MODEL.CHECKPOINT_FILE_NAME = os.path.basename(best_cp_path)

    if cfg.DPR.DO_TEST:
        config.DPR.DATA.TEST_DATA_PATH = os.path.join(data_path, 'mixed', 'norm_test.json')
        model_file = get_model_file(cfg, cfg.DPR.MODEL.CHECKPOINT_FILE_NAME)
        retriever_trainer = RetrieverTrainer(cfg, model_file=model_file)
        test_dataset = RetDataset(
            file=cfg.DPR.DATA.TEST_DATA_PATH,
            num_pos_ctx=None,
            num_total_ctx=None,
            normalize=cfg.DPR.DATA.NORMALIZE,
            flatten_attr=cfg.DPR.DATA.FLATTEN_ATTRIBUTE,
            is_train=False
        )
        result_list = retriever_trainer.evaluate(test_dataset)
        ranking_result_path = os.path.join(cfg.OUTPUT_PATH, 'rank_score_ids.jsonl')
        save_ranking_results(result_list, ranking_result_path)
        logger.info('Rank and score saved in %s', ranking_result_path)
        combined_result_path = os.path.join(cfg.OUTPUT_PATH, 'combined_score_ids.json')
        save_combined_results(result_list, config.DPR.DATA.TEST_DATA_PATH, combined_result_path)
        logger.info('Combined score saved in %s', combined_result_path)
        eval_metrics = ["map", "r-precision", "mrr", "ndcg", "hit_rate@5", "precision@1",
                        "hits@5", "precision@3", "precision@5", "map@1", "map@3",
                        "map@5", "recall@1", "recall@3", "recall@5", "f1@1", "f1@3", "f1@5"]
        metrics_dt = compute_metrics(result_list, eval_metrics, comp_separate=True)
        eval_metrics_path = os.path.join(cfg.OUTPUT_PATH, f'eval_metrics')
        save_eval_metrics(metrics_dt, eval_metrics_path)
        logger.info('Evaluation done. Score per metric saved in %s', eval_metrics_path)


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    exp_cfg_path = arguments['--path_cfg_exp']
    data_path = arguments['--path_data']
    model_path = arguments['--path_model']
    output_path = arguments['--path_output']
    dpr_ckpt = arguments['--dpr_ckpt']
    version = arguments['--version']
    config = get_cfg_defaults()

    logger.info("Started logging...")
    if exp_cfg_path is not None:
        config.merge_from_file(exp_cfg_path)
    if data_path is not None:
        config.DPR.DATA.DATA_PATH = data_path
    if output_path is not None:
        config.OUTPUT_PATH = output_path
    if dpr_ckpt is not None:
        config.DPR.MODEL.CHECKPOINT_FILE_NAME = dpr_ckpt
    if version is None:
        version = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
        logger.info(f"Version: {version}")

    # Make result folders if they do not exist
    config.OUTPUT_PATH = os.path.join(config.OUTPUT_PATH, config.EXP, version)
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH, exist_ok=False)
    print(f'Output path: {config.OUTPUT_PATH}')
    logger.info(f'Output path: {config.OUTPUT_PATH}')
    if model_path is not None:
        config.DPR.MODEL.MODEL_PATH = model_path
    else:
        config.DPR.MODEL.MODEL_PATH = config.OUTPUT_PATH
    print(f'Model path: {config.DPR.MODEL.MODEL_PATH}')
    logger.info(f'Model path: {config.DPR.MODEL.MODEL_PATH}')
    run(config)
    shutil.copy(src='dpr_logs.log', dst=os.path.join(config.OUTPUT_PATH, f'dpr_logs_{datetime.now().strftime("%d_%m_%Y-%H_%M_%S")}.log'))
