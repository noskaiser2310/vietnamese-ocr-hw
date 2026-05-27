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

import editdistance

VDIAC = set('àáâãăảạấầẩẫậắằẳẵặèéêếềểễệìíỉĩịòóôõơởờỡợùúũưứừửữựỳỷỹỵđÀÁÂÃĂẢẠẤẦẨẪẬẮẰẲẴẶÈÉÊẾỀỂỄỆÌÍỈĨỊÒÓÔÕƠỞỜỠỢÙÚŨƯỨỪỬỮỰỲỶỸỴĐ')

def calculate_metrics(preds, targets):
    total_cer, total_wer, total_dcer = 0, 0, 0
    total_chars, total_words, total_diac = 0, 0, 0
    
    for pred, target in zip(preds, targets):
        # CER
        total_cer += editdistance.eval(pred, target)
        total_chars += max(1, len(target))
        
        # WER
        pred_words = pred.split()
        target_words = target.split()
        total_wer += editdistance.eval(pred_words, target_words)
        total_words += max(1, len(target_words))
        
        # D-CER (Diacritic Character Error Rate)
        pd = [c for c in pred if c in VDIAC]
        gd = [c for c in target if c in VDIAC]
        total_dcer += editdistance.eval(pd, gd)
        total_diac += max(1, len(gd))
        
    return total_cer, total_chars, total_wer, total_words, total_dcer, total_diac
