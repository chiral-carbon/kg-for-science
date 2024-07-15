from typing import Dict


def classify_predictions(gold: dict, pred: dict, union=False) -> Dict[str, float]:
    """
    Returns true positives, false positives, and false negatives for one example
    If union is True, then disregards the type of the tag and only considers the union of all tags
    """
    n_tp = 0
    n_fp = 0
    n_fn = 0
    if union:
        gold_phrases = set(phrase for phrases in gold.values() for phrase in phrases)
        pred_phrases = set(phrase for phrases in pred.values() for phrase in phrases)
        n_tp = len(gold_phrases & pred_phrases)
        n_fp = len(pred_phrases - gold_phrases)
        n_fn = len(gold_phrases - pred_phrases)
        return n_tp, n_fp, n_fn

    for tag in set(gold.keys()).union(pred.keys()):
        gold_phrases = set(gold.get(tag, []))
        pred_phrases = set(pred.get(tag, []))

        n_tp += len(gold_phrases & pred_phrases)
        n_fp += len(pred_phrases - gold_phrases)
        n_fn += len(gold_phrases - pred_phrases)

    return n_tp, n_fp, n_fn


def compute_metrics(running_time, pred_times, runtype, eval_metrics=None):
    metrics = {}
    metrics["avg_pred_response_time_per_sentence"] = (
        round(sum(pred_times) / len(pred_times), 4) if pred_times else 0
    )
    metrics["total_time"] = round(running_time, 4)

    if runtype == "eval" and eval_metrics is not None:
        n_tp, n_fp, n_fn, n_tp_union, n_fp_union, n_fn_union = eval_metrics

        precision = round(n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0, 4)
        recall = round(n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0, 4)
        f1 = round(
            (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            ),
            4,
        )
        union_precision = round(
            (
                n_tp_union / (n_tp_union + n_fp_union)
                if (n_tp_union + n_fp_union) > 0
                else 0
            ),
            4,
        )
        union_recall = round(
            (
                n_tp_union / (n_tp_union + n_fn_union)
                if (n_tp_union + n_fn_union) > 0
                else 0
            ),
            4,
        )
        union_f1 = round(
            (
                2 * (union_precision * union_recall) / (union_precision + union_recall)
                if (union_precision + union_recall) > 0
                else 0
            ),
            4,
        )

        metrics.update(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "union_precision": union_precision,
                "union_recall": union_recall,
                "union_f1": union_f1,
            }
        )

    return metrics
