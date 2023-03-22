import csv
import os
import json
import argparse
import logging
import pickle
import pandas as pd
from typing import List, Tuple, Dict, Iterator
from datetime import datetime

import numpy as np
import torch
from torch import Tensor as T
import torch.nn as nn
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import load_states_from_checkpoint, get_model_obj, get_model_file
from dpr.models.hf_models import get_bert_biencoder_components
from dpr.indexer.faiss_indexers import DenseIndexer, DenseFlatIndexer
from dpr.options import set_encoder_params_from_state

from dpr.utils.data_utils import normalize_attr_passage

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


class Ranker:
    def __init__(self, q_encoder: nn.Module, tensorizer: Tensorizer, indexer: DenseIndexer):
        self.q_encoder = q_encoder
        self.tensorizer = tensorizer
        self.indexer = indexer

    def embed_questions(self, rows: List[str], args) -> T:
        n_rows = len(rows)
        bsz = args.batch_size
        embeds = []
        for batch_idx, batch_start in enumerate(range(0, n_rows, bsz)):
            batch_end = batch_start + bsz
            batch_rows = rows[batch_start:batch_end]
            # Assuming last element of batch row is question string
            batch_token_tensors = [self.tensorizer.text_to_tensor(batch_row[-1]) for batch_row in batch_rows]

            ids_batch = torch.stack(batch_token_tensors, dim=0).to(args.device)
            seg_batch = torch.zeros_like(ids_batch).to(args.device)
            attn_mask = self.tensorizer.get_attn_mask(ids_batch).to(args.device)
            with torch.no_grad():
                _, out, _ = self.q_encoder(ids_batch, seg_batch, attn_mask)

            embeds.extend(out.cpu().split(1, dim=0))
            # if len(embeds) % 100 == 0:
            #     logger.info('Encoded queries %d', len(embeds))

        embeds = torch.cat(embeds, dim=0)
        assert embeds.size(0) == n_rows
        return embeds

    def index_embeds(self, embed_data: List[Tuple[int, List[float]]], buffer_size: int = 500):
        buffer = []
        for cid, ctx_embed in embed_data:
            buffer.append((cid, np.array(ctx_embed)))
            if buffer_size == len(buffer):
                self.indexer.index_data(buffer)
                buffer = []
        self.indexer.index_data(buffer)

    def get_top_docs(self, q_embeds: np.array, top_docs: int=10) -> List[Tuple[List[object], List[float]]]:
        results = self.indexer.search_knn(q_embeds, top_docs)
        return results


def get_text_embeds(args, texts, tensorizer, encoder):
    '''
    Encodes texts into embeddings
    :param args:
    :param texts: List of Strings to encode
    :param tensorizer: Object to tokenize and tensorize texts
    :param encoder: Model to encode texts
    :return: List of embedding of texts
    '''
    n_texts = len(texts)
    bsz = args.batch_size
    embeds = []
    for batch_idx, batch_start in enumerate(range(0, n_texts, bsz)):
        batch_end = batch_start + bsz
        batch_texts = texts[batch_start:batch_end]
        batch_token_tensors = [tensorizer.text_to_tensor(text) for text in batch_texts]

        ids_batch = torch.stack(batch_token_tensors, dim=0).to(args.device)
        seg_batch = torch.zeros_like(ids_batch).to(args.device)
        attn_mask = tensorizer.get_attn_mask(ids_batch).to(args.device)
        with torch.no_grad():
            _, out, _ = encoder(ids_batch, seg_batch, attn_mask)
        embeds.extend(out.cpu().split(1, dim=0))
    embeds = torch.cat(embeds, dim=0)
    assert embeds.size(0) == n_texts
    return embeds


def load_encoder_components(args, load_ctx_encoder=False, load_question_encoder=False):
    ckpt_state = None
    model_file = get_model_file(args, args.dpr_checkpoint_file_name)
    if model_file:
        ckpt_state = load_states_from_checkpoint(model_file)
        # Not sure if needed. It overwrites args params from checkpoint states
        set_encoder_params_from_state(ckpt_state.encoder_params, args)
    tensorizer, biencoder, _ = get_bert_biencoder_components(args, inference_only=True)
    ctx_encoder, question_encoder = None, None
    if load_ctx_encoder:
        ctx_encoder = biencoder.ctx_model
        ctx_encoder.to(args.device)
    if load_question_encoder:
        question_encoder = biencoder.question_model
        question_encoder.to(args.device)
    if ckpt_state:
        logger.info('Loading saved model state ...')
        logger.debug('saved model keys =%s', ckpt_state.model_dict.keys())
        if load_ctx_encoder and ctx_encoder:
            ctx_encoder_prefix = 'ctx_model.'
            prefix_len = len(ctx_encoder_prefix)
            ctx_state = {key[prefix_len:]: value for (key, value) in ckpt_state.model_dict.items() if
                         key.startswith(ctx_encoder_prefix)}

            get_model_obj(ctx_encoder).load_state_dict(ctx_state)
        if load_question_encoder and question_encoder:
            question_encoder_prefix = 'question_model.'
            prefix_len = len(question_encoder_prefix)
            question_state = {key[prefix_len:]: value for (key, value) in ckpt_state.model_dict.items() if
                         key.startswith(question_encoder_prefix)}

            get_model_obj(question_encoder).load_state_dict(question_state)
    return tensorizer, ctx_encoder, question_encoder


def get_normalized_evidences(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    df = df[['cid', 'source', 'candidate']].drop_duplicates()
    df['norm_cand'] = df.apply(lambda row: normalize_attr_passage(row['candidate']), axis=1)
    return df[['cid', 'norm_cand']].values


def embed_contexts(args):
    tensorizer, ctx_encoder, _ = load_encoder_components(args, load_ctx_encoder=True)
    ctx_encoder.eval()
    # Collect context documents and embed them
    ctx_rows = get_normalized_evidences(args.ctx_path)
    ctx_embeds = get_text_embeds(args, ctx_rows[:8], tensorizer, ctx_encoder)
    return ctx_embeds


def evaluate_ranker_with_shared_ctx(args):
    if args.embed_ctx:
        ctx_embeds = embed_contexts(args)
    else:
        ctx_embeds = pickle.load(args.ctx_embed_path)

    tensorizer, _, q_encoder = load_encoder_components(args, load_question_encoder=True)
    indexer = DenseFlatIndexer(q_encoder.get_out_size())
    ranker = Ranker(q_encoder, tensorizer, indexer)
    ranker.index_embeds(ctx_embeds, buffer_size=args.index_buffer_size)


def precision_recall_at_k(ranked_items, k):
    all_recalls = []
    all_precisions = []
    for _, _, pred_ctx_ids, act_ctx_ids in ranked_items:
        if len(pred_ctx_ids) >= k:
            act_set = set(act_ctx_ids)
            pred_set = set(pred_ctx_ids[:k])
            true_positive = (act_set & pred_set)
            cur_recall = len(true_positive) / len(act_set)
            all_recalls.append(cur_recall)
            cur_precision = len(true_positive) / len(pred_set)
            all_precisions.append(cur_precision)

    p_count = len(all_precisions)
    mean_precision = sum(all_precisions) / p_count if p_count else None
    r_count = len(all_recalls)
    mean_recall = sum(all_recalls) / r_count if r_count else None
    return mean_precision, p_count, mean_recall, r_count


def recall_at_k_hit_miss(ranked_items, k):
    total, hits = 0, 0
    for _, _, pred_ctx_ids, act_ctx_ids in ranked_items:
        # assert(k <= len(pred_ctx_ids))
        if len(pred_ctx_ids) >= k:
            total += 1
            act_set = set(act_ctx_ids)
            pred_set = set(pred_ctx_ids[:k])
            true_positive = (act_set & pred_set)
            if len(true_positive) > 0:
                hits += 1
    mean_recall = hits/total if total else None
    return mean_recall, total


def precision_at_r(ranked_items):
    all_precisions = []
    for _, _, pred_ctx_ids, act_ctx_ids in ranked_items:
        # print(pred_ctx_ids, act_ctx_ids)
        r = len(act_ctx_ids)
        act_set = set(act_ctx_ids)
        pred_set = set(pred_ctx_ids[:r])
        true_positive = (act_set & pred_set)
        # print(true_positive)
        cur_precision = len(true_positive) / len(pred_set)
        all_precisions.append(cur_precision)
    count = len(all_precisions)
    mean_precision_at_r = sum(all_precisions) / count if count else None
    return mean_precision_at_r, count


def MRR(ranked_items):
    reciprocals = []
    for _, _, pred_ctx_ids, act_ctx_ids in ranked_items:
        true_positive = (set(act_ctx_ids) & set(pred_ctx_ids))
        if len(true_positive):
            first_rank = min([pred_ctx_ids.index(idx) for idx in true_positive]) + 1
            reciprocal = 1 / first_rank
        else:
            reciprocal = 0
        reciprocals.append(reciprocal)
    count = len(reciprocals)
    mean_rr = sum(reciprocals)/count if count else None
    return mean_rr, count


def MAP_at_k(ranked_items, k):
    ap_at_k = []
    for _, _, pred_ctx_ids, act_ctx_ids in ranked_items:
        if len(pred_ctx_ids) >= k:
            ap = 0
            for x in range(k):
                act_set = set(act_ctx_ids)
                pred_set = set(pred_ctx_ids[:x+1])
                if pred_ctx_ids[x] in act_set:
                    true_positive = (act_set & pred_set)
                    prec_at_k = len(true_positive) / (x+1)
                    ap += prec_at_k
            ap_q = ap / len(act_ctx_ids)
            ap_at_k.append(ap_q)
    count = len(ap_at_k)
    mean_ap = sum(ap_at_k)/count if count else None
    return mean_ap, count


def compute_eval_scores(ranked_items, ks=None):
    if ks is None:
        ks = [1, 3, 5, 10]
    metrics = {'Precision_at_r': None,
               'MRR': None,
               'MAP_at_k': {},
               'Precision_at_k': {},
               'Recall_at_k': {},
               'Recall_at_k_hit_miss': {}}
    prec_at_r, prec_at_r_count = precision_at_r(ranked_items)
    metrics['Precision_at_r'] = {'val': prec_at_r, 'count': prec_at_r_count}
    mrr, mrr_count = MRR(ranked_items)
    metrics['MRR'] = {'val': mrr, 'count': mrr_count}
    for k in ks:
        map_k, map_k_count = MAP_at_k(ranked_items, k)
        metrics['MAP_at_k'][k] = {'val': map_k, 'count': map_k_count}
        pk, p_count, rk, r_count = precision_recall_at_k(ranked_items, k)
        metrics['Precision_at_k'][k] = {'val': pk, 'count': p_count}
        metrics['Recall_at_k'][k] = {'val': rk, 'count': r_count}
        recall_hm, recall_hm_count = recall_at_k_hit_miss(ranked_items, k)
        metrics['Recall_at_k_hit_miss'][k] = {'val': recall_hm, 'count': recall_hm_count}
    return metrics


def retrank_single_case(group, tensorizer, ctx_encoder, q_encoder, top_ctx_count):
    indexer = DenseFlatIndexer(ctx_encoder.get_out_size())
    ctx_embeds = get_text_embeds(args, group['ctxs'], tensorizer, ctx_encoder)
    ctx_embeds_arr = np.array(ctx_embeds)
    cids_arr = np.array(group['cids'])
    assert len(ctx_embeds_arr) == len(cids_arr)
    indexer.index_data(ctx_embeds_arr, cids_arr)

    q_embeds = get_text_embeds(args, [group['question']], tensorizer, q_encoder)
    q_embeds_arr = np.array(q_embeds)
    scores_arr, ctx_ids_arr = indexer.search_knn(q_embeds_arr, min(top_ctx_count, len(group['cids'])))
    return scores_arr[0], ctx_ids_arr[0]


def save_ranking_results(result_list, ranking_result_path):
    with open(ranking_result_path, 'w') as fout:
        names = ['qid', 'scores', 'pred_ctx_ids', 'actual_ctx_ids']
        for vals in result_list:
            dt = dict(zip(names, vals))
            json_line = json.dumps(dt)
            fout.write(json_line+'\n')


def save_eval_metrics(metrics_dt, eval_metrics_path):
    with open(eval_metrics_path, 'w') as fout:
        json.dump(metrics_dt, fout, indent=4)


def evaluate_ranker_wo_shared_ctx(args):
    tensorizer, ctx_encoder, q_encoder = load_encoder_components(args, load_ctx_encoder=True, load_question_encoder=True)
    ctx_encoder.eval()
    q_encoder.eval()
    result_list = []

    eval_file_path = os.path.join(args.data_dir, 'evidence_ranking_dev.csv')
    with open(eval_file_path, 'r') as fin:
        reader = csv.reader(fin, delimiter='\t')
        header = next(reader)
        group = {'qid': None}
        for row in reader:
            row = dict(zip(header, row))
            if row['qid'] != group['qid']:
                if len(result_list) == 5:
                    break
                if group['qid'] and group['vcids']:
                    # Process the group for retrieval
                    ctx_scores_arr, ctx_ids_arr = retrank_single_case(group, tensorizer, ctx_encoder, q_encoder, args.top_ctx_count)
                    result_list.append((group['qid'], ctx_scores_arr.tolist(), ctx_ids_arr.tolist(), group['vcids']))
                # Start collecting new group
                group = {'qid': None, 'question': None, 'ctxs': [], 'cids': [], 'vcids': [], 'sources': []}
                group['qid'] = row['qid']
                group['question'] = row['question']
            group['ctxs'].append(normalize_attr_passage(row['candidate']))
            group['cids'].append(int(row['cid']))
            group['sources'].append(row['source'])
            if int(row['label']):
                group['vcids'].append(int(row['cid']))
        if group['qid'] and group['vcids']:
            # Process the last group for retrieval
            _, ctx_ids_arr = retrank_single_case(group, tensorizer, ctx_encoder, q_encoder, args.top_ctx_count)
            result_list.append((group['qid'], ctx_scores_arr.tolist(), ctx_ids_arr.tolist(), group['vcids']))

    date_time = datetime.now().strftime("%d_%m_%Y-%H_%M_%S")
    ranking_result_path = os.path.join(cfg.OUTPUT_PATH, f'rank_score_ids{date_time}.jsonl')
    save_ranking_results(result_list, ranking_result_path)
    metrics_dt = compute_eval_scores(result_list)
    eval_metrics_path = os.path.join(cfg.OUTPUT_PATH, f'eval_metrics{date_time}.json')
    save_eval_metrics(metrics_dt, eval_metrics_path)
    return metrics_dt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./data/',
                        help="Data Directory")
    parser.add_argument('--output_dir', type=str, default='./output/',
                        help="Output Directory")
    parser.add_argument('--top_ctx_count', type=int, default=10,
                        help="Amount of top docs to return")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for training")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument('--index_buffer_size', type=int, default=500,
                        help="Temporal memory data buffer size (in samples) for indexer")

    parser.add_argument("--pretrained_model_cfg", default='bert-base-uncased', type=str,
                        help="config name for model initialization")
    parser.add_argument("--encoder_model_type", default='ance_roberta', type=str,
                        help="model type. One of [hf_bert, pytext_bert, fairseq_roberta]")
    parser.add_argument("--projection_dim", default=0, type=int,
                        help="Extra linear layer on top of standard bert/roberta encoder")
    parser.add_argument("--sequence_length", type=int, default=512,
                        help="Max length of the encoder input sequence")
    parser.add_argument('--dpr_checkpoint_file_name', type=str, default=None,
                        help="dpr checkpoint")

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Device: {args.device}')
    if not os.path.exists(cfg.OUTPUT_PATH):
        os.mkdir(cfg.OUTPUT_PATH)
    eval_metrics = evaluate_ranker_wo_shared_ctx(args)
    eval_metrics_json = json.dumps(eval_metrics, indent=4)
    print(eval_metrics_json)
