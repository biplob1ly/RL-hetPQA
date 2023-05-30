"""
train model
Usage:
    rag_trainer.py  --path_cfg_exp=<path> [--path_data=<path>] [--path_model=<path>] [--path_output=<path>] [--version=<val>] [--rag_ckpt=<filename>]
    rag_trainer.py -h | --help

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
    --rag_ckpt=<filename>       RAG checkpoint file name
"""
from docopt import docopt
import os
import shutil
import time
from datetime import datetime
import numpy as np
import torch
from torch import Tensor as T
import logging
import random
from typing import Tuple
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from configs.rag_config.config import get_cfg_defaults

from rag.utils.model_utils import (
    get_checkpoint_path, get_model_components,
    get_optimizer_components, setup_for_distributed_mode,
    load_states_from_checkpoint, CheckpointState, get_model_obj,
    set_model_cfg_from_state, get_model_params_state
)
from rag.utils.data_utils import GenDataset, GenCollator
from rag.options import setup_cfg_gpu, set_seed
from rag_utils import BLEUScorer, RAGValResult, format_rag_validation, save_combined_results, save_eval_metrics, delete

logging.basicConfig(
    filename='rag_logs.log',
    filemode='w',
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGTrainer:
    def __init__(self, cfg, checkpoint_path=None):
        self.cfg = cfg
        self.shard_id = cfg.LOCAL_RANK if cfg.LOCAL_RANK != -1 else 0
        self.distributed_factor = cfg.DISTRIBUTED_WORLD_SIZE or 1
        saved_state = None
        if checkpoint_path:
            saved_state = load_states_from_checkpoint(checkpoint_path)
            set_model_cfg_from_state(saved_state.model_params, cfg)
        tokenizer, generator = get_model_components(cfg, checkpoint_path)
        optimizer, scheduler = get_optimizer_components(cfg, generator)
        generator, optimizer = setup_for_distributed_mode(generator, optimizer, cfg.DEVICE, cfg.N_GPU,
                                                      cfg.LOCAL_RANK,
                                                      cfg.FP16,
                                                      cfg.FP16_OPT_LEVEL)
        self.tokenizer = tokenizer
        self.generator = generator
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.start_step = 0
        self.scheduler_state = None
        self.validations = []
        self.saved_cps = {}
        self.best_cp_name = None
        self.train_dataset = None
        self.val_dataset = None
        self.collator = GenCollator(tokenizer, cfg.RAG.MODEL.PROMPT_MAX_LENGTH, cfg.RAG.MODEL.ANSWER_MAX_LENGTH)
        self.eval_scorer = BLEUScorer()
        if saved_state:
            self._load_saved_state(saved_state)

    def evaluate(self, eval_dataset: GenDataset):
        logger.info('Evaluating generator ...')
        self.generator.eval()
        cfg = self.cfg
        eval_sampler = SequentialSampler(eval_dataset)
        eval_data_loader = DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=cfg.RAG.SOLVER.TEST_BATCH_SIZE,
            drop_last=False,
            num_workers=1,
            collate_fn=self.collator
        )
        bleu_scores = []
        result_data = []
        with torch.no_grad():
            for iteration, batch in enumerate(eval_data_loader):
                model_outputs = self.generator.generate(
                    input_ids=batch.prompt_ids.to(cfg.DEVICE),
                    attention_mask=batch.prompt_masks.to(cfg.DEVICE),
                    max_length=cfg.RAG.SOLVER.EVAL_ANSWER_MAX_LEN
                )
                for i, out_seq in enumerate(model_outputs):
                    pred_answer = self.tokenizer.decode(out_seq, skip_special_tokens=True)
                    data_example = eval_dataset.get_example(batch.indices[i])
                    gold_answers = data_example['answers']
                    score = self.eval_scorer.compute_bleu_score(gold_answers, pred_answer)
                    bleu_scores.append(score)
                    data_example['pred_answer'] = {'text': pred_answer, 'bleu': score}
                    result_data.append(data_example)
        mean_bleu_score = np.mean(bleu_scores)
        return mean_bleu_score, result_data

    def _save_checkpoint(self, step: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.generator)
        cp_dir = os.path.join(cfg.RAG.MODEL.MODEL_PATH, cfg.RAG.MODEL.CHECKPOINT_FILE_NAME + '.' + str(step))
        os.makedirs(cp_dir, exist_ok=True)
        model_to_save.save_pretrained(cp_dir)
        cp_fp = os.path.join(cp_dir, "checkpoint.pth.tar")

        meta_params = get_model_params_state(cfg)
        state = CheckpointState(
            meta_params,
            self.optimizer.state_dict(),
            self.scheduler.state_dict(),
            step
        )
        torch.save(state._asdict(), cp_fp)
        logger.info('Saved checkpoint at %s', cp_fp)
        return cp_dir

    def validate_and_save(self, cur_step: int, val_dataset: GenDataset):
        cfg = self.cfg
        # for distributed mode, save checkpoint for only one process
        save_cp = cfg.LOCAL_RANK in [-1, 0]

        cur_val_id = len(self.validations)
        if cfg.RAG.DATA.VAL_DATA_PATH:
            mean_bleu_score, _ = self.evaluate(val_dataset)
            val_metrics = ["bleu"]
            metrics_score = [mean_bleu_score]
            rag_eval = RAGValResult(cur_val_id, cur_step, val_metrics, metrics_score)
            self.validations.append(rag_eval)
            fmt_header, fmt_value = format_rag_validation(rag_eval)
            logger.info(fmt_header)
            logger.info(fmt_value)
            if cur_val_id == 0:
                print(fmt_header)
            print(fmt_value)

        if save_cp:
            best_rag_eval = max(self.validations, key=lambda x: x.scores)
            if len(self.saved_cps) < cfg.RAG.SOLVER.CP_SAVE_LIMIT:
                cp_path = self._save_checkpoint(cur_step)
                self.saved_cps[cur_val_id] = cp_path
                if best_rag_eval.val_id == cur_val_id:
                    self.best_cp_name = cp_path
                    logger.info('New Best validation checkpoint %s', cp_path)
            else:
                sorted_rag_evals = sorted(self.validations, key=lambda x: x.scores, reverse=True)
                for rag_eval in sorted_rag_evals[cfg.RAG.SOLVER.CP_SAVE_LIMIT:]:
                    if rag_eval.val_id in self.saved_cps:
                        delete(self.saved_cps[rag_eval.val_id])
                        del self.saved_cps[rag_eval.val_id]
                        cp_path = self._save_checkpoint(cur_step)
                        self.saved_cps[cur_val_id] = cp_path
                        if best_rag_eval.val_id == cur_val_id:
                            self.best_cp_name = cp_path
                            logger.info('New Best validation checkpoint %s', cp_path)
                        break

    def train(self, train_dataset, val_dataset=None):
        self.generator.train()
        cfg = self.cfg
        train_sampler = RandomSampler(train_dataset)
        train_data_loader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=cfg.RAG.SOLVER.TRAIN_BATCH_SIZE,
            drop_last=True,
            num_workers=1,
            collate_fn=self.collator
        )

        logger.info("Total updates=%d", cfg.RAG.SOLVER.TOTAL_TRAIN_STEPS)
        logger.info(" Eval step = %d", cfg.RAG.SOLVER.NUM_STEP_PER_EVAL)
        logger.info("***** Training *****")
        cur_step = self.start_step
        rolling_loss = 0
        epoch = 0
        last_saved_step = -1
        while cur_step < cfg.RAG.SOLVER.TOTAL_TRAIN_STEPS:
            epoch += 1
            logger.info("***** Epoch %d *****", epoch)
            for iteration, batch in enumerate(train_data_loader):
                model_outputs = self.generator(
                    input_ids=batch.prompt_ids.to(cfg.DEVICE),
                    attention_mask=batch.prompt_masks.to(cfg.DEVICE),
                    labels=batch.target_ids.to(cfg.DEVICE)
                )
                cur_loss = model_outputs.loss
                if self.cfg.RAG.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS > 1:
                    cur_loss = cur_loss / self.cfg.RAG.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS
                rolling_loss += cur_loss.item()
                cur_loss.backward()
                if (iteration + 1) % self.cfg.RAG.SOLVER.OPTIMIZER.GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), cfg.RAG.SOLVER.OPTIMIZER.MAX_GRAD_NORM)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.generator.zero_grad()
                    cur_step += 1

                if cur_step % cfg.RAG.SOLVER.NUM_STEP_PER_EVAL == 0 and last_saved_step != cur_step:
                    logger.info(
                        "Rank=%d, step: %d/%d, avg train loss: %f, lr: %f",
                        cfg.LOCAL_RANK,
                        cur_step,
                        cfg.RAG.SOLVER.TOTAL_TRAIN_STEPS,
                        rolling_loss/cfg.RAG.SOLVER.NUM_STEP_PER_EVAL,
                        self.scheduler.get_last_lr()[0]
                    )
                    self.validate_and_save(cur_step, val_dataset)
                    self.generator.train()
                    rolling_loss = 0
                    last_saved_step = cur_step
                if cur_step >= cfg.RAG.SOLVER.TOTAL_TRAIN_STEPS:
                    break

        logger.info(
            "Rank=%d, step: %d/%d, avg train loss: %f, lr: %f",
            cfg.LOCAL_RANK,
            cur_step,
            cfg.RAG.SOLVER.TOTAL_TRAIN_STEPS,
            rolling_loss / cfg.RAG.SOLVER.NUM_STEP_PER_EVAL,
            self.scheduler.get_last_lr()[0]
        )
        self.validate_and_save(cur_step, val_dataset)
        logger.info("********** Training Completed **********")
        if cfg.LOCAL_RANK in [-1, 0]:
            for idx, rag_val_result in enumerate(self.validations):
                fmt_header, fmt_value = format_rag_validation(rag_val_result)
                if idx == 0:
                    logger.info(fmt_header)
                logger.info(fmt_value)
            logger.info("Training finished. Best validation checkpoint %s", self.best_cp_name)
        return self.best_cp_name

    def _load_saved_state(self, saved_state: CheckpointState):
        if self.cfg.RAG.SOLVER.RESET_CHECKPOINT_STEP:
            self.step = 0
        else:
            self.step = saved_state.step

        if not self.cfg.RAG.SOLVER.OPTIMIZER.RESET:
            if saved_state.optimizer_dict:
                logger.info('Loading saved optimizer state ...')
                self.optimizer.load_state_dict(saved_state.optimizer_dict)
            if saved_state.scheduler_dict:
                logger.info("Loading scheduler state %s", saved_state.scheduler_dict)
                self.scheduler.load_state_dict(saved_state.scheduler_dict)


def run(cfg):
    cfg = setup_cfg_gpu(cfg)
    set_seed(cfg.SEED)
    logger.info("***** Initializing model components *****")
    if cfg.RAG.DO_TRAIN:
        config.RAG.DATA.TRAIN_DATA_PATH = os.path.join(data_path, 'mixed', 'train.json')
        config.RAG.DATA.VAL_DATA_PATH = os.path.join(data_path, 'mixed', 'dev.json')
        checkpoint_path = get_checkpoint_path(cfg, cfg.RAG.MODEL.CHECKPOINT_FILE_NAME)
        rag_trainer = RAGTrainer(cfg, checkpoint_path=checkpoint_path)
        train_dataset = GenDataset(
            cfg.RAG.DATA.TRAIN_DATA_PATH,
            n_context=cfg.RAG.DATA.NUM_CONTEXT,
            normalize=cfg.RAG.DATA.NORMALIZE,
            flatten_attr=cfg.RAG.DATA.FLATTEN_ATTRIBUTE
        )
        val_dataset = GenDataset(
            cfg.RAG.DATA.VAL_DATA_PATH,
            n_context=cfg.RAG.DATA.NUM_CONTEXT,
            normalize=cfg.RAG.DATA.NORMALIZE,
            flatten_attr=cfg.RAG.DATA.FLATTEN_ATTRIBUTE
        )
        best_cp_path = rag_trainer.train(train_dataset, val_dataset=val_dataset)
        cfg.dump(stream=open(os.path.join(cfg.RAG.MODEL.MODEL_PATH, f'config_{cfg.EXP}.yaml'), 'w'))
        cfg.RAG.MODEL.CHECKPOINT_FILE_NAME = os.path.basename(best_cp_path)

    if cfg.RAG.DO_TEST:
        config.RAG.DATA.TEST_DATA_PATH = os.path.join(data_path, 'mixed', 'test.json')
        checkpoint_path = get_checkpoint_path(cfg, cfg.RAG.MODEL.CHECKPOINT_FILE_NAME)
        rag_trainer = RAGTrainer(cfg, checkpoint_path=checkpoint_path)
        test_dataset = GenDataset(
            cfg.RAG.DATA.TEST_DATA_PATH,
            n_context=cfg.RAG.DATA.NUM_CONTEXT,
            normalize=cfg.RAG.DATA.NORMALIZE,
            flatten_attr=cfg.RAG.DATA.FLATTEN_ATTRIBUTE
        )
        mean_bleu_score, result_data = rag_trainer.evaluate(test_dataset)
        combined_result_path = os.path.join(cfg.OUTPUT_PATH, 'combined_result.json')
        save_combined_results(result_data, combined_result_path)
        logger.info('Combined score saved in %s', combined_result_path)
        metrics_dt = {'BLEU': mean_bleu_score}
        eval_metrics_path = os.path.join(cfg.OUTPUT_PATH, f'eval_metrics')
        save_eval_metrics(metrics_dt, eval_metrics_path)
        logger.info('Evaluation done. Score per metric saved in %s', eval_metrics_path)


if __name__ == "__main__":
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    exp_cfg_path = arguments['--path_cfg_exp']
    data_path = arguments['--path_data']
    model_path = arguments['--path_model']
    output_path = arguments['--path_output']
    rag_ckpt = arguments['--rag_ckpt']
    version = arguments['--version']
    config = get_cfg_defaults()

    logger.info("Started logging...")
    if exp_cfg_path is not None:
        config.merge_from_file(exp_cfg_path)
    if data_path is not None:
        config.RAG.DATA.DATA_PATH = data_path
    if output_path is not None:
        config.OUTPUT_PATH = output_path
    if rag_ckpt is not None:
        config.RAG.MODEL.CHECKPOINT_FILE_NAME = rag_ckpt
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
        config.RAG.MODEL.MODEL_PATH = model_path
    else:
        config.RAG.MODEL.MODEL_PATH = config.OUTPUT_PATH
    print(f'Model path: {config.RAG.MODEL.MODEL_PATH}')
    logger.info(f'Model path: {config.RAG.MODEL.MODEL_PATH}')
    run(config)
    shutil.copy(src='rag_logs.log', dst=os.path.join(config.OUTPUT_PATH, f'rag_logs_{datetime.now().strftime("%d_%m_%Y-%H_%M_%S")}.log'))
