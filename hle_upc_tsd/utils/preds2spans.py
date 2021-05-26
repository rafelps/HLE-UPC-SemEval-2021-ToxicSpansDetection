def preds2spans(preds, masks, offsets):
    """
    :param preds: tensor (batch, seq_len)
    :param masks: tensor (batch, seq_len)
    :param offsets: tensor (batch, seq_len, 2)
    :returns spans: list[set] (batch, spans)
    """
    out_spans = []
    for sent_id in range(preds.shape[0]):
        n_tokens = masks[sent_id].sum()
        token_preds = preds[sent_id, 1:n_tokens + 1]
        token_offsets = offsets[sent_id, 1:n_tokens + 1]

        pred_spans = set()
        for token_id in range(n_tokens):
            # Add token's offsets
            if token_preds[token_id] == 1:
                pred_spans.update(range(token_offsets[token_id, 0], token_offsets[token_id, 1]))
                # Add offsets between two toxic tokens (whitespaces, hyphens, ...)
                if token_id != 0 and token_preds[token_id - 1] == 1:
                    pred_spans.update(range(token_offsets[token_id - 1, 1], token_offsets[token_id, 0]))
        pred_spans = pred_spans - {-1}

        out_spans.append(pred_spans)
    return out_spans
