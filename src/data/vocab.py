import json
import os

class Vocab:
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            d = json.load(f)
            self.c2i = d.get('char_to_id', d)
            
        self.i2c = {int(v): k for k, v in self.c2i.items()}
        self.PAD = self.c2i.get('<pad>', 0)
        self.SOS = self.c2i.get('<s>', 1)
        self.EOS = self.c2i.get('</s>', 2)
        self.UNK = self.c2i.get('<unk>', 3)
        self.V = len(self.c2i)
        
    def encode(self, text, max_len=90):
        ids = [self.SOS] + [self.c2i.get(c, self.UNK) for c in text] + [self.EOS]
        if len(ids) > max_len:
            ids = ids[:max_len-1] + [self.EOS]
        return ids
        
    def decode(self, ids):
        chars = []
        for i in ids:
            if i in (self.SOS, self.PAD): continue
            if i == self.EOS: break
            chars.append(self.i2c.get(i, '?'))
        return ''.join(chars)
