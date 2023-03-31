import collections
import json
import pandas as pd
from ranx import Qrels, Run, evaluate, compare


DPRTrainerRun = collections.namedtuple(
    'DPRTrainerRun',
    [
        "run_id",
        "epoch",
        "iteration",
        "val_loss",
        "metrics",
        "scores"
    ]
)


def format_dpr_run(dpr_run: DPRTrainerRun):
    header = ['run_id', 'epoch', 'iteration', 'val_loss'] + dpr_run.metrics
    fmt_header = ' | '.join([f"{item:->10}" for item in header])
    values = [dpr_run.run_id, dpr_run.epoch, dpr_run.iteration, dpr_run.val_loss] + dpr_run.scores
    fmt_value = f"{values[0]: >10}" + ' | ' + ' | '.join([f"{item: >10.5f}" for item in values[1:]])
    return fmt_header, fmt_value


def save_ranking_results(result_list, ranking_result_path):
    with open(ranking_result_path, 'w') as fout:
        for val_dt in result_list:
            json_line = json.dumps(val_dt)
            fout.write(json_line+'\n')


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


def compute_metrics(result_list, metrics):
    sources = ['all', "attribute", "bullet", "OSP", "Desc", "review", "CQA"]
    qrels_dt = {source: {} for source in sources}
    run_dt = {source: {} for source in sources}
    count = 0
    for q_dt in result_list:
        ctx_act_score = {str(ctx_id): 1 for ctx_id in q_dt['actual_ctx_ids']}
        ctx_pred_score = {str(ctx_id): score for ctx_id, score in zip(q_dt['pred_ctx_ids'], q_dt['scores'])}

        unq_qid = q_dt['qid'] + '_' + str(count)
        qrels_dt[q_dt['source']][unq_qid] = ctx_act_score
        run_dt[q_dt['source']][unq_qid] = ctx_pred_score

        qrels_dt['all'][unq_qid] = ctx_act_score
        run_dt['all'][unq_qid] = ctx_pred_score
        count += 1

    score_dict = {}
    for source in sources:
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


def compute_eval_scores(ranked_items, ks=None):
    if ks is None:
        ks = [1, 3, 5]
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