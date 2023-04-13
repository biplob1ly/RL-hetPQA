import collections
import json
import pandas as pd
from ranx import Qrels, Run, evaluate, compare
import copy


DPRTrainerRun = collections.namedtuple(
    'DPRTrainerRun',
    [
        "val_id",
        "epoch",
        "iteration",
        "val_loss",
        "metrics",
        "scores"
    ]
)


def format_dpr_run(dpr_run: DPRTrainerRun):
    header = ['val_id', 'epoch', 'iteration', 'val_loss'] + dpr_run.metrics
    fmt_header = ' | '.join([f"{item:->12}" for item in header])
    values = [dpr_run.val_id, dpr_run.epoch, dpr_run.iteration, dpr_run.val_loss] + dpr_run.scores
    fmt_value = ' | '.join([f"{item: >12}" for item in values[:3]]) + ' | ' + ' | '.join([f"{item: >12.5f}" for item in values[3:]])
    return fmt_header, fmt_value


def save_ranking_results(result_list, ranking_result_path):
    with open(ranking_result_path, 'w') as fout:
        for val_dt in result_list:
            json_line = json.dumps(val_dt)
            fout.write(json_line+'\n')


def save_combined_results(result_list, test_data_path, combined_result_path):
    with open(test_data_path, 'r') as fin:
        test_samples = json.load(fin)
    test_samples = [r for r in test_samples if len(r["positive_ctxs"]) > 0]
    for sample, result in zip(test_samples, result_list):
        assert sample['qid'] == result['qid']
        ctx_pred_score = {ctx_id: score for ctx_id, score in zip(result['pred_ctx_ids'], result['scores'])}
        all_ctxs = copy.deepcopy(sample['positive_ctxs'] + sample['negative_ctxs'])
        del sample['negative_ctxs']
        for ctx in all_ctxs:
            ctx['dp'] = ctx_pred_score[ctx['cid']]
        sample['pred_ctxs'] = sorted(all_ctxs, key=lambda x: x['dp'], reverse=True)
    with open(combined_result_path, 'w') as fout:
        json.dump(test_samples, fout, indent=4)


def save_eval_metrics(metrics_dt, eval_metrics_path):
    with open(eval_metrics_path + '.json', 'w') as fout:
        json.dump(metrics_dt, fout, indent=4)

    col_dt = collections.defaultdict(list)
    for source, dt in metrics_dt.items():
        col_dt['source'].append(source)
        for metric, score in dt.items():
            col_dt[metric].append(score)
    df = pd.DataFrame(col_dt)
    with open(eval_metrics_path + '.csv', 'w') as fout:
        df.to_csv(fout, index=False)


def compute_metrics(result_list, metrics, comp_separate=False):
    qrels_dt = collections.defaultdict(dict)
    run_dt = collections.defaultdict(dict)
    count = 0
    for q_dt in result_list:
        ctx_act_score = {ctx_id: 1 for ctx_id in q_dt['actual_ctx_ids']}
        ctx_pred_score = {ctx_id: score for ctx_id, score in zip(q_dt['pred_ctx_ids'], q_dt['scores'])}

        unq_qid = q_dt['qid'] + '_' + str(count)
        if comp_separate:
            ctx_source = {ctx_id: source for ctx_id, source in zip(q_dt['pred_ctx_ids'], q_dt['pred_ctx_sources'])}
            pos_sources = set([ctx_source[ctx_id] for ctx_id in q_dt['actual_ctx_ids']])
            for source in pos_sources:
                qrels_dt[source][unq_qid] = {ctx_id: rel for ctx_id, rel in ctx_act_score.items() if ctx_source[ctx_id]==source}
                run_dt[source][unq_qid] = {ctx_id: score for ctx_id, score in ctx_pred_score.items() if ctx_source[ctx_id]==source}

        qrels_dt['all'][unq_qid] = ctx_act_score
        run_dt['all'][unq_qid] = ctx_pred_score
        count += 1

    score_dict = {}
    for source in qrels_dt:
        qrels = Qrels(qrels_dt[source])
        run = Run(run_dt[source])
        score_dict[source] = evaluate(qrels, run, metrics)
        score_dict[source]['count'] = len(qrels_dt[source])
    return score_dict


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
        r = len(act_ctx_ids)
        act_set = set(act_ctx_ids)
        pred_set = set(pred_ctx_ids[:r])
        true_positive = (act_set & pred_set)
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
            ap = []
            act_set = set(act_ctx_ids)
            for x in range(k):
                pred_set = set(pred_ctx_ids[:x+1])
                if pred_ctx_ids[x] in act_set:
                    true_positive = (act_set & pred_set)
                    prec_at_k = len(true_positive) / (x+1)
                    ap.append(prec_at_k)
            ap_q = sum(ap) / len(act_ctx_ids)
            ap_at_k.append(ap_q)
    count = len(ap_at_k)
    mean_ap = sum(ap_at_k)/count if count else None
    return mean_ap, count