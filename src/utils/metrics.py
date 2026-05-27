def ctc_decode(logp, vocab, blank=0):
    """
    Greedy Decoder cho CTC.
    logp: shape (T, B, C) hoac (B, T, C). Can chuyen ve (B, T, C) roi thuc hien Argmax.
    """
    if logp.shape[0] == logp.shape[1] or logp.shape[-1] == vocab.V:
        # Neu truyen vao la (T, B, C) -> chuyen thanh (B, T)
        preds = logp.argmax(-1)
        if preds.shape[0] != logp.shape[1]: # Differentiate T and B
             preds = preds.permute(1, 0)
    else:
        preds = logp.argmax(-1)

    out = []
    for seq in preds:
        chars, prev = [], blank
        for i in seq.tolist():
            if i != blank and i != prev:
                chars.append(vocab.i2c.get(i, '?'))
            prev = i
        out.append(''.join(chars))
    return out
