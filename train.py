import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.crnn import CRNN
from src.data.dataset import ViHTRDataset, AspectGroupedBatchSampler
from src.data.vocab import Vocab
from src.utils.metrics import ctc_decode
import editdistance

def calculate_metrics(preds, targets):
    total_cer, total_wer, total_chars, total_words = 0, 0, 0, 0
    
    for pred, target in zip(preds, targets):
        # CER
        cer_dist = editdistance.eval(pred, target)
        total_cer += cer_dist
        total_chars += len(target)
        
        # WER
        pred_words = pred.split()
        target_words = target.split()
        wer_dist = editdistance.eval(pred_words, target_words)
        total_wer += wer_dist
        total_words += len(target_words)
        
    return total_cer, total_chars, total_wer, total_words

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # 1. Load Data
    print("[INFO] Preparing Data...")
    vocab = Vocab(args.vocab)
    
    train_dataset = ViHTRDataset(args.data_dir, args.train_file, vocab, is_train=True)
    val_dataset = ViHTRDataset(args.data_dir, args.val_file, vocab, is_train=False)
    
    train_sampler = AspectGroupedBatchSampler(train_dataset, args.batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=train_dataset.collate_fn, num_workers=4)
    
    val_sampler = AspectGroupedBatchSampler(val_dataset, args.batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=val_dataset.collate_fn, num_workers=4)
    
    # 2. Model & Loss & Optimizer
    model = CRNN(vocab.V).to(device)
    print(f"[INFO] Model parameters: {model.n_params():,}")
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_cer = float('inf')
    
    # 3. Training Loop
    print("[INFO] Starting Training...")
    for ep in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}")
        
        for batch in pbar:
            imgs = batch['images'].to(device)
            targets = batch['targets'].to(device)
            target_lens = batch['target_lens'].to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logp, input_lens = model(imgs)
                loss = criterion(logp, targets, input_lens, target_lens)
                
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        total_cer_dist, total_cer_chars = 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['images'].to(device)
                targets = batch['targets'].to(device)
                target_lens = batch['target_lens'].to(device)
                
                with torch.cuda.amp.autocast():
                    logp, input_lens = model(imgs)
                    loss = criterion(logp, targets, input_lens, target_lens)
                    
                val_loss += loss.item()
                preds = ctc_decode(logp, vocab, blank=0)
                
                # Convert target IDs to text
                tgt_texts = []
                idx = 0
                for length in target_lens:
                    tgt_texts.append(vocab.decode(targets[idx:idx+length].tolist()))
                    idx += length
                    
                cer_dist, chars, _, _ = calculate_metrics(preds, tgt_texts)
                total_cer_dist += cer_dist
                total_cer_chars += chars
                
        val_loss /= len(val_loader)
        val_cer = total_cer_dist / total_cer_chars * 100
        
        print(f"Epoch {ep} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | CER: {val_cer:.2f}%")
        
        if val_cer < best_cer:
            best_cer = val_cer
            torch.save(model.state_dict(), args.save_path)
            print(f">>> Saved new best model to {args.save_path} (CER: {best_cer:.2f}%)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--val_file', type=str, required=True)
    parser.add_argument('--vocab', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='crnn_best.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()
    
    train(args)
