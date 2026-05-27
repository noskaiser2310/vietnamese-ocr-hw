import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv

class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden=256, n=2, drop=0.2):
        super().__init__()
        r = tv.resnet34(weights=tv.ResNet34_Weights.IMAGENET1K_V1)
        # ResNet34 Stem: output shape (B, 512, H/32, W/32)
        self.cnn = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        # AdaptiveAvgPool2d to collapse height dimension to 1
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        
        # RNN Decoder
        self.rnn = nn.LSTM(256, hidden, n, batch_first=True, bidirectional=True, dropout=drop if n>1 else 0.)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hidden*2, vocab_size)

    def forward(self, imgs):
        # imgs: (B, 3, 128, W)
        f = self.cnn(imgs) # (B, 256, 8, W/16)
        
        # Pool Height to 1, then reshape to (B, T, C)
        f = self.pool(f).squeeze(2).permute(0, 2, 1)
        
        # Sequence modeling
        o, _ = self.rnn(f)
        lp = F.log_softmax(self.fc(self.drop(o)), -1)
        
        T, B = lp.shape[1], lp.shape[0]
        # PyTorch CTCLoss expects shape (T, B, C)
        return lp.permute(1, 0, 2), torch.full((B,), T, dtype=torch.long)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
