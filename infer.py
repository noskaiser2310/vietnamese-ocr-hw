import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image

from src.models.crnn import CRNN
from src.data.vocab import Vocab
from src.utils.metrics import ctc_decode

def infer(img_path, ckpt_path, vocab_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return
        
    vocab = Vocab(vocab_path)
    model = CRNN(vocab.V).to(device)
    
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        return
        
    print(f"[INFO] Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    if 'state' in ckpt:
        model.load_state_dict(ckpt['state'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    nw = max(128, min(int(128 * w / h), 2048))
    img = img.resize((nw, 128), Image.Resampling.LANCZOS)
    
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    tensor = tf(img).unsqueeze(0).to(device)
    
    print("[INFO] Running inference...")
    with torch.no_grad():
        lp, _ = model(tensor)
        
        # logp is (T, B, C)
        preds = ctc_decode(lp, vocab, blank=0)
        
    print("=" * 40)
    try:
        print(f"Predicted Text: {preds[0]}")
    except UnicodeEncodeError:
        print(f"Predicted Text (bytes): {preds[0].encode('utf-8')}")
    print("=" * 40)
    
    return preds[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='Path to image')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint (.pt)')
    parser.add_argument('--vocab', type=str, required=True, help='Path to vocab.json')
    args = parser.parse_args()
    
    infer(args.img, args.ckpt, args.vocab)
