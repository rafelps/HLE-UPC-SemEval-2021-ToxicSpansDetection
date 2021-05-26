def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1. if len(predictions) == 0 else 0.
    nom = 2 * len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions)) + len(set(gold))
    return nom / denom


def f1_score(preds, targets, reduce=True):
    """
    Mean batch f1-score
    :param preds: list (batch, spans)
    :param targets: tensor (batch, seq_len)
    :param reduce: If True it returns all individual metrics
    """
    f1s = []
    for sent_id in range(len(preds)):
        predicted_spans = preds[sent_id]
        target_spans = set(targets[sent_id].tolist()) - {-1}
        f1s.append(f1(predicted_spans, target_spans))
    if not reduce:
        return f1s
    return sum(f1s) / len(f1s)
