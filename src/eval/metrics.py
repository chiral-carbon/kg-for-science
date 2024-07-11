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
