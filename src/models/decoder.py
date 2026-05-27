import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding to inject sequence order.
    """
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [Seq_Len, Batch, Dim]
        return x + self.pe[:x.size(0)]


class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive Transformer Decoder to predict Vietnamese character sequences.
    Expects standard [Seq_Len, Batch, Embed_Dim] shapes for PyTorch Transformer compatibility.
    """
    def __init__(self, vocab_size, embed_dim=512, num_layers=6, num_heads=8, max_seq_len=200):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Sinusoidal positional encoding
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)
        
        # Transformer Decoder Stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4,
            dropout=0.1
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output classification layer
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
    def generate_square_subsequent_mask(self, sz, device):
        """
        Generates an upper-triangular causal mask to prevent decoder from looking at future tokens.
        """
        mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt (Tensor): Target tokens [Seq_Tgt, Batch]
            memory (Tensor): Encoded features [Seq_Memory, Batch, Embed_Dim] (outputs from DACA)
            tgt_mask (Tensor, optional): Causal mask [Seq_Tgt, Seq_Tgt]
            memory_mask (Tensor, optional): Encoder-Decoder cross-attention mask
        Returns:
            logits (Tensor): Predicted character logits [Seq_Tgt, Batch, Vocab_Size]
        """
        # 1. Embed target tokens and apply positional encoding
        tgt_embed = self.embedding(tgt)  # [Seq_Tgt, Batch, Embed_Dim]
        tgt_embed = self.pos_encoder(tgt_embed)
        
        # 2. If causal mask is not provided, generate it
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0), tgt.device)
            
        # 3. Transformer Decoder pass
        out = self.transformer_decoder(
            tgt=tgt_embed, 
            memory=memory, 
            tgt_mask=tgt_mask, 
            memory_mask=memory_mask
        )  # [Seq_Tgt, Batch, Embed_Dim]
        
        # 4. Project to vocabulary size logits
        logits = self.fc_out(out)  # [Seq_Tgt, Batch, Vocab_Size]
        
        return logits
