import torch
import torch.nn as nn
from .encoders import GlobalContextEncoder, LocalDiacriticsDetector
from .attention import DiacriticsAwareCrossAttention
from .decoder import AutoregressiveDecoder

class DAHTRModel(nn.Module):
    """
    DA-HTR: Diacritics-Aware Handwritten Text Recognition Model
    Fuses Global Context and High-Res Local features using DACA Layer 
    and decodes character sequences autoregressively.
    """
    def __init__(self, vocab_size, global_dim=768, local_dim=256, embed_dim=512, backbone_name="resnet34"):
        super().__init__()
        self.global_encoder = GlobalContextEncoder(backbone_name=backbone_name, global_dim=global_dim)
        self.local_detector = LocalDiacriticsDetector(backbone_name=backbone_name, feature_dim=local_dim)
        self.daca = DiacriticsAwareCrossAttention(dim_q=global_dim, dim_k=local_dim, dim_out=embed_dim)
        self.decoder = AutoregressiveDecoder(vocab_size=vocab_size, embed_dim=embed_dim)
        
    def forward(self, img, tgt, tgt_mask=None):
        """
        Args:
            img (Tensor): Input images [Batch, 3, Height, Width]
            tgt (Tensor): Target tokens [Seq_Tgt, Batch]
            tgt_mask (Tensor, optional): Causal target mask
        Returns:
            logits (Tensor): Predicted character logits [Seq_Tgt, Batch, Vocab_Size]
            attn_map (Tensor): Local spatial attention maps [Batch, 1, Height_Local, Width_Local]
        """
        # 1. Global features [Batch, Seq_Q, Global_Dim]
        g_feat = self.global_encoder(img)
        
        # 2. Local high-res features [Batch, Seq_K, Local_Dim]
        l_feat, attn_map = self.local_detector(img)
        
        # 3. Fused representation [Batch, Seq_Q, Embed_Dim]
        fused = self.daca(g_feat, l_feat)
        
        # Convert shapes for Transformer Decoder (expected Seq_Len, Batch, Embed_Dim)
        memory = fused.permute(1, 0, 2)
        
        # 4. Decode text
        logits = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return logits, attn_map

    @torch.no_grad()
    def generate(self, img, max_length=90, sos_token_id=1, eos_token_id=2, pad_token_id=0):
        """
        Greedy Search decoding algorithm for text line prediction (inference / validation).
        Args:
            img (Tensor): Input images [Batch, 3, Height, Width]
            max_length (int): Maximum generation sequence length
            sos_token_id (int): Start of sequence token ID
            eos_token_id (int): End of sequence token ID
            pad_token_id (int): Padding token ID
        Returns:
            predictions (Tensor): Generated character token IDs [Batch, Seq_Len]
            attn_map (Tensor): Local spatial attention maps [Batch, 1, Height_Local, Width_Local]
        """
        self.eval()
        
        # 1. Extract and fuse features
        g_feat = self.global_encoder(img)
        l_feat, attn_map = self.local_detector(img)
        fused = self.daca(g_feat, l_feat)
        memory = fused.permute(1, 0, 2)  # [Seq_Memory, Batch, Embed_Dim]
        
        batch_size = img.size(0)
        device = img.device
        
        # 2. Initialize target tensor with <s> token
        ys = torch.full((1, batch_size), sos_token_id, dtype=torch.long, device=device)  # [Seq_Len, Batch]
        
        # Track finished batches
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i in range(max_length - 1):
            # Target causal mask is handled internally in decoder
            logits = self.decoder(ys, memory)  # [Seq_Tgt, Batch, Vocab_Size]
            
            # Extract logits of the last token generated
            next_token_logits = logits[-1]  # [Batch, Vocab_Size]
            next_tokens = torch.argmax(next_token_logits, dim=-1)  # [Batch]
            
            # If batch item is already finished, replace with pad token
            next_tokens = torch.where(finished, torch.tensor(pad_token_id, device=device), next_tokens)
            
            # Update finished status
            finished |= (next_tokens == eos_token_id)
            
            # Append predicted tokens
            ys = torch.cat([ys, next_tokens.unsqueeze(0)], dim=0)  # [Seq_Len + 1, Batch]
            
            if finished.all():
                break
                
        # Permute back to [Batch, Seq_Len]
        return ys.permute(1, 0), attn_map
