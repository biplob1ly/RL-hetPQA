"""
train model
Usage:
    dpr_trainer.py --path_output=<path> --path_data=<path> [--path_train_data=<path>] [--path_val_data=<path>] [--path_cfg_exp=<path>]
    dpr_trainer.py -h | --help

Options:
    -h --help                   show this screen help
    --path_output=<path>        output path
    --path_data=<path>          data path
    --path_train_data=<path>    train data path
    --path_val_data=<path>      validation data path
    --path_cfg_exp=<path>       experiment config path
"""
from docopt import docopt
import os
import shutil
import time
import heapq
from datetime import datetime
import math
import torch
from torch import Tensor as T
import logging
import random
from typing import Tuple
from configs.config import get_cfg_defaults

from dpr.models.hf_models import get_bert_biencoder_components
from dpr.models.biencoder import BiEncoder, BiEncoderNllLoss, BiEncoderBatch
from dpr.utils.model_utils import (
    load_states_from_checkpoint, get_model_obj,
    setup_for_distributed_mode, CheckpointState,
    get_schedule_linear, move_to_device, get_model_file
)
from dpr.options import set_encoder_params_from_state, get_encoder_params_state, setup_cfg_gpu, set_seed
from dpr.utils.data_utils import JsonQADataset, SharedDataIterator

logging.basicConfig(
    filename='logs.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class RetrieverTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.shard_id = cfg.LOCAL_RANK if cfg.LOCAL_RANK != -1 else 0
        self.distributed_factor = cfg.DISTRIBUTED_WORLD_SIZE or 1

        logger.info("***** Initializing components for training *****")
        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(cfg, cfg.DPR.MODEL.CHECKPOINT_FILE_NAME)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            # Not sure if needed. It overwrites cfg params from checkpoint states
            set_encoder_params_from_state(saved_state.encoder_params, cfg)

        tensorizer, biencoder, optimizer = get_bert_biencoder_components(cfg)
        model, optimizer = setup_for_distributed_mode(biencoder, optimizer, cfg.DEVICE, cfg.N_GPU,
                                                      cfg.LOCAL_RANK,
                                                      cfg.FP16,
                                                      cfg.FP16_OPT_LEVEL)
        self.biencoder = biencoder
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.top_cp_scores = []
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        self.train_dataset = JsonQADataset(cfg.DPR.DATA.TRAIN_DATA_PATH,
                                           normalize=cfg.DPR.DATA.NORMALIZE,
                                           flatten_attr=cfg.DPR.DATA.FLATTEN_ATTRIBUTE)
        self.val_dataset = JsonQADataset(cfg.DPR.DATA.VAL_DATA_PATH)
        self.loss_function = BiEncoderNllLoss()
        if saved_state:
            strict = True if cfg.DPR.MODEL.PROJECTION_DIM == 0 else False
            self._load_saved_state(saved_state, strict=strict)
        self.dev_iterator = None

    def get_data_iterator(
        self,
        batch_size: int,
        is_train_set: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0
    ):
        dataset = self.train_dataset if is_train_set else self.val_dataset
        dataset.load_data()
        logger.info("Initializing %s iterator, size: %d", 'training' if is_train_set else 'validation', len(dataset))
        return SharedDataIterator(
            dataset=dataset,
            shard_id=self.shard_id,
            num_shards=self.distributed_factor,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            offset=offset
        )

    def _calc_loss(
            self,
            local_q_vector,
            local_ctx_vectors,
            local_positive_idxs,
            local_hard_negatives_idxs: list = None,
    ) -> Tuple[T, int]:
        """
        Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
        across all the nodes.
        """
        cfg = self.cfg
        distributed_world_size = cfg.DISTRIBUTED_WORLD_SIZE or 1
        if distributed_world_size > 1:
            raise NotImplementedError
        else:
            global_q_vector = local_q_vector
            global_ctxs_vector = local_ctx_vectors
            positive_idx_per_question = local_positive_idxs
            hard_negatives_per_question = local_hard_negatives_idxs

        loss, correct_predictions_count = self.loss_function.calc(
            global_q_vector,
            global_ctxs_vector,
            positive_idx_per_question,
            hard_negatives_per_question
        )
        return loss, correct_predictions_count

    def _do_biencoder_fwd_pass(
            self,
            batch_input: BiEncoderBatch
    ) -> Tuple[torch.Tensor, int]:
        batch_input = BiEncoderBatch(**move_to_device(batch_input._asdict(), self.cfg.DEVICE))

        q_attn_mask = self.tensorizer.get_attn_mask(batch_input.question_ids)
        ctx_attn_mask = self.tensorizer.get_attn_mask(batch_input.context_ids)

        if self.biencoder.training:
            model_out = self.biencoder(batch_input.question_ids, batch_input.question_segments, q_attn_mask,
                                       batch_input.context_ids, batch_input.ctx_segments, ctx_attn_mask)
        else:
            with torch.no_grad():
                model_out = self.biencoder(batch_input.question_ids, batch_input.question_segments, q_attn_mask,
                                           batch_input.context_ids, batch_input.ctx_segments, ctx_attn_mask)

        local_q_vector, local_ctx_vectors = model_out
        loss, correct_predictions_count = self._calc_loss(local_q_vector, local_ctx_vectors,
                                                          batch_input.positive_ctx_indices,
                                                          batch_input.hard_neg_ctx_indices)

        total_correct_preds = correct_predictions_count.sum().item()

        if self.cfg.N_GPU > 1:
            loss = loss.mean()
        if self.cfg.DPR.SOLVER.GRADIENT_ACCUMULATION_STEPS > 1:
            loss = loss / self.cfg.DPR.SOLVER.GRADIENT_ACCUMULATION_STEPS

        return loss, total_correct_preds

    def validate_average_rank(self) -> float:
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info('Average rank validation ...')

        cfg = self.cfg
        self.biencoder.eval()
        distributed_factor = self.distributed_factor
        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                batch_size=cfg.DPR.SOLVER.VAL_BATCH_SIZE,
                is_train_set=False,
                shuffle=False
            )
        data_iterator = self.dev_iterator

        sub_batch_size = cfg.DPR.SOLVER.VAL_AV_RANK_BSZ
        sim_score_f = BiEncoderNllLoss.get_similarity_function()
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        num_hard_negatives = cfg.DPR.SOLVER.VAL_AV_RANK_HARD_NEG
        num_other_negatives = cfg.DPR.SOLVER.VAL_AV_RANK_OTHER_NEG

        log_result_step = cfg.DPR.SOLVER.LOG_BATCH_STEP

        for i, samples_batch in enumerate(data_iterator.iterate_data()):
            if len(q_represenations) > cfg.DPR.SOLVER.VAL_AV_RANK_MAX_QS / distributed_factor:
                break
            biencoder_input = BiEncoder.create_biencoder_input(
                samples=samples_batch,
                tensorizer=self.tensorizer,
                insert_title=True,
                num_hard_negatives=num_hard_negatives,
                num_other_negatives=num_other_negatives,
                shuffle=False
            )
            biencoder_input = BiEncoderBatch(**move_to_device(biencoder_input._asdict(), cfg.DEVICE))
            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            bsz = ctxs_ids.size(0)

            # split contexts batch into sub batches since it is supposed to be too large to be processed in one batch
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):

                q_ids, q_segments = (
                    (biencoder_input.question_ids, biencoder_input.question_segments) if j == 0 else (None, None)
                )

                if j == 0 and cfg.N_GPU > 1 and q_ids.size(0) == 1:
                    # if we are in DP (but not in DDP) mode, all model input tensors should have batch size >1 or 0,
                    # otherwise the other input tensors will be split but only the first split will be called
                    continue

                ctx_ids_batch = ctxs_ids[batch_start:batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[batch_start:batch_start + sub_batch_size]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)
                with torch.no_grad():
                    q_dense, ctx_dense = self.biencoder(
                        q_ids,
                        q_segments,
                        q_attn_mask,
                        ctx_ids_batch,
                        ctx_seg_batch,
                        ctx_attn_mask
                    )

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.positive_ctx_indices
            positive_idx_per_question.extend([total_ctxs + v for v in batch_positive_idxs])

            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Av.rank validation: step %d, computed ctx_vectors %d, q_vectors %d",
                    i,
                    len(ctx_represenations),
                    len(q_represenations),
                )

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        q_represenations = torch.cat(q_represenations, dim=0)

        logger.info('Av.rank validation: total q_vectors size=%s', q_represenations.size())
        logger.info('Av.rank validation: total ctx_vectors size=%s', ctx_represenations.size())

        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        scores = sim_score_f(q_represenations, ctx_represenations)
        values, indices = torch.sort(scores, dim=1, descending=True)

        rank = 0
        for i, idx in enumerate(positive_idx_per_question):
            # aggregate the rank of the known gold passage in the sorted results for each question
            gold_idx = (indices[i] == idx).nonzero()
            rank += gold_idx.item()

        if distributed_factor > 1:
           raise NotImplementedError

        av_rank = float(rank / q_num)
        logger.info('Av.rank validation: average rank %s, total questions=%d', av_rank, q_num)
        return av_rank

    def validate_nll(self) -> float:
        logger.info('NLL validation ...')
        cfg = self.cfg
        self.biencoder.eval()
        if not self.dev_iterator:
            self.dev_iterator = self.get_data_iterator(
                batch_size=cfg.DPR.SOLVER.VAL_BATCH_SIZE,
                is_train_set=False,
                shuffle=False
            )
        data_iterator = self.dev_iterator
        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        num_hard_negatives = cfg.DPR.DATA.HARD_NEGATIVES
        num_other_negatives = cfg.DPR.DATA.OTHER_NEGATIVES
        log_result_step = cfg.DPR.SOLVER.LOG_BATCH_STEP
        batches = 0
        for i, samples_batch in enumerate(data_iterator.iterate_data()):
            biencoder_batch = BiEncoder.create_biencoder_input(
                samples=samples_batch,
                tensorizer=self.tensorizer,
                insert_title=True,
                num_hard_negatives=num_hard_negatives,
                num_other_negatives=num_other_negatives,
                shuffle=False
            )

            loss, correct_cnt = self._do_biencoder_fwd_pass(biencoder_batch)
            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1) % log_result_step == 0:
                logger.info('Eval step: %d , used_time=%f sec., loss=%f ', i, time.time() - start_time, loss.item())

        total_loss = total_loss / batches
        total_samples = batches * cfg.DPR.SOLVER.VAL_BATCH_SIZE * self.distributed_factor
        correct_ratio = float(total_correct_predictions / total_samples)
        logger.info(
            "NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f",
            total_loss,
            total_correct_predictions,
            total_samples,
            correct_ratio,
        )
        return total_loss

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(cfg.OUTPUT_PATH,
                          cfg.DPR.MODEL.CHECKPOINT_FILE_NAME + '.' + str(epoch) + ('.' + str(offset) if offset > 0 else ''))

        meta_params = get_encoder_params_state(cfg)

        state = CheckpointState(
            model_to_save.state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params
        )
        torch.save(state._asdict(), cp)
        logger.info('Saved checkpoint at %s', cp)
        return cp

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.LOCAL_RANK in [-1, 0]

        # if epoch == cfg.DPR.SOLVER.VAL_AV_RANK_START_EPOCH:
        #     self.best_validation_result = None

        if cfg.DPR.DATA.VAL_DATA_PATH:
            if epoch >= cfg.DPR.SOLVER.VAL_AV_RANK_START_EPOCH:
                validation_loss = self.validate_average_rank()
            else:
                validation_loss = self.validate_nll()
        else:
            validation_loss = 0

        if save_cp:
            if len(self.top_cp_scores) < cfg.DPR.SOLVER.CP_SAVE_LIMIT:
                cp_path = self._save_checkpoint(scheduler, epoch, iteration)
                heapq.heappush(self.top_cp_scores, (-validation_loss, cp_path))
                if max(self.top_cp_scores)[1] == cp_path:
                    logger.info('New Best validation checkpoint %s', cp_path)
            elif validation_loss < -self.top_cp_scores[0][0]:
                _, cp_path_to_delete = heapq.heappop(self.top_cp_scores)
                os.remove(cp_path_to_delete)
                cp_path = self._save_checkpoint(scheduler, epoch, iteration)
                heapq.heappush(self.top_cp_scores, (-validation_loss, cp_path))
                if max(self.top_cp_scores)[1] == cp_path:
                    logger.info('New Best validation checkpoint %s', cp_path)

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        eval_step: int,
        train_data_iterator: SharedDataIterator
    ):
        cfg = self.cfg
        rolling_train_loss = 0.0
        epoch_loss = 0
        epoch_correct_predictions = 0

        log_result_step = cfg.DPR.SOLVER.LOG_BATCH_STEP
        rolling_loss_step = cfg.DPR.SOLVER.TRAIN_ROLLING_LOSS_STEP
        num_hard_negatives = cfg.DPR.DATA.HARD_NEGATIVES
        num_other_negatives = cfg.DPR.DATA.OTHER_NEGATIVES
        seed = cfg.SEED
        self.biencoder.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0

        # TODO: Check this
        # biencoder = get_model_obj(self.biencoder)
        for i, samples_batch in enumerate(train_data_iterator.iterate_data(epoch=epoch)):

            # to be able to resume shuffled ctx- pools
            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)

            batch_input = BiEncoder.create_biencoder_input(
                samples=samples_batch,
                tensorizer=self.tensorizer,
                insert_title=True,
                num_hard_negatives=num_hard_negatives,
                num_other_negatives=num_other_negatives,
                shuffle=True,
                shuffle_positives=self.train_dataset.shuffle_positives
            )

            loss, correct_cnt = self._do_biencoder_fwd_pass(batch_input)
            epoch_correct_predictions += correct_cnt
            epoch_loss += loss.item()
            rolling_train_loss += loss.item()

            if cfg.FP16:
                from apex import amp
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                if cfg.DPR.SOLVER.MAX_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), cfg.DPR.SOLVER.MAX_GRAD_NORM)
            else:
                loss.backward()
                if cfg.DPR.SOLVER.MAX_GRAD_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(self.biencoder.parameters(), cfg.DPR.SOLVER.MAX_GRAD_NORM)

            if (i + 1) % cfg.DPR.SOLVER.GRADIENT_ACCUMULATION_STEPS == 0:
                self.optimizer.step()
                scheduler.step()
                self.biencoder.zero_grad()

            if i % log_result_step == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    "Epoch: %d: Step: %d/%d, loss=%f, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    loss.item(),
                    lr,
                )

            if (i + 1) % rolling_loss_step == 0:
                logger.info("Train batch %d", data_iteration)
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                logger.info(
                    "Avg. loss per last %d batches: %f",
                    rolling_loss_step,
                    latest_rolling_train_av_loss,
                )
                rolling_train_loss = 0.0

            if data_iteration % eval_step == 0:
                logger.info(
                    "rank=%d, Validation: Epoch: %d Step: %d/%d",
                    cfg.LOCAL_RANK,
                    epoch,
                    data_iteration,
                    epoch_batches,
                )
                self.validate_and_save(epoch, train_data_iterator.get_iteration(), scheduler)
                self.biencoder.train()

        logger.info("Epoch finished on rank %d", cfg.LOCAL_RANK)
        self.validate_and_save(epoch, data_iteration, scheduler)

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        logger.info("epoch total correct predictions=%d", epoch_correct_predictions)

    def train(self):
        cfg = self.cfg
        train_iterator = self.get_data_iterator(
            batch_size=cfg.DPR.SOLVER.TRAIN_BATCH_SIZE,
            is_train_set=True,
            shuffle=True,
            shuffle_seed=cfg.SEED,
            offset=self.start_batch
        )
        max_iterations = train_iterator.max_iterations
        logger.info("  Total iterations per epoch=%d", max_iterations)
        if max_iterations == 0:
            logger.warning("No data found for training.")
            return

        updates_per_epoch = train_iterator.max_iterations // cfg.DPR.SOLVER.GRADIENT_ACCUMULATION_STEPS
        total_updates = updates_per_epoch * cfg.DPR.SOLVER.NUM_TRAIN_EPOCH
        logger.info(" Total updates=%d", total_updates)
        warmup_steps = cfg.DPR.SOLVER.WARMUP_STEPS
        scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)
        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            scheduler.load_state_dict(self.scheduler_state)

        eval_step = math.ceil(updates_per_epoch / cfg.DPR.SOLVER.EVAL_PER_EPOCH)
        logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        for epoch in range(self.start_epoch, int(cfg.DPR.SOLVER.NUM_TRAIN_EPOCH)):
            logger.info("***** Epoch %d *****", epoch)
            self._train_epoch(scheduler, epoch, eval_step, train_iterator)

        if cfg.LOCAL_RANK in [-1, 0]:
            logger.info('validation loss\tcp_path ')
            for neg_val_loss, cp_path in sorted(self.top_cp_scores, reverse=True):
                print(f'{-neg_val_loss:<15.5f}\t{cp_path}')
                logger.info(f'{-neg_val_loss:<15.5f}\t{cp_path}')
            logger.info("Training finished. Best validation checkpoint %s", self.best_cp_name)

    def _load_saved_state(self, saved_state: CheckpointState, strict=True):
        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info('Loading checkpoint @ batch=%s and epoch=%s', offset, epoch)

        self.start_epoch = epoch
        self.start_batch = offset

        model_to_load = get_model_obj(self.biencoder)
        logger.info('Loading saved model state ...')
        model_to_load.load_state_dict(saved_state.model_dict, strict=strict)

        if self.cfg.DPR.SOLVER.OPTIMIZER.RESET:
            pass
        else:
            if saved_state.optimizer_dict:
                logger.info('Loading saved optimizer state ...')
                self.optimizer.load_state_dict(saved_state.optimizer_dict)

            if saved_state.scheduler_dict:
                self.scheduler_state = saved_state.scheduler_dict


def run(cfg):
    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg.SEED)
    retriever_trainer = RetrieverTrainer(cfg)
    retriever_trainer.train()


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    output_path = arguments['--path_output']
    data_path = arguments['--path_data']
    train_data_path = arguments['--path_train_data']
    val_data_path = arguments['--path_val_data']
    exp_cfg_path = arguments['--path_cfg_exp']
    config = get_cfg_defaults()
    if data_path is not None:
        config.DPR.DATA.DATA_PATH = data_path
        config.DPR.DATA.TRAIN_DATA_PATH = os.path.join(data_path, 'train.json')
        config.DPR.DATA.VAL_DATA_PATH = os.path.join(data_path, 'dev.json')
    if train_data_path is not None:
        config.DPR.DATA.TRAIN_DATA_PATH = train_data_path
    if val_data_path is not None:
        config.DPR.DATA.VAL_DATA_PATH = val_data_path
    if exp_cfg_path is not None:
        config.merge_from_file(exp_cfg_path)
    if output_path is not None:
        config.OUTPUT_PATH = output_path

    # Make result folders if they do not exist
    cur_timestamp = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    config.OUTPUT_PATH = os.path.join(config.OUTPUT_PATH, config.EXP, cur_timestamp)
    print(config.OUTPUT_PATH)
    if not os.path.exists(config.OUTPUT_PATH):
        os.makedirs(config.OUTPUT_PATH, exist_ok=False)
    config.dump(stream=open(os.path.join(config.OUTPUT_PATH, f'config_{config.EXP}.yaml'), 'w'))
    logger.info("Started logging...")
    run(config)
    shutil.copy(src='logs.log', dst=config.OUTPUT_PATH)
