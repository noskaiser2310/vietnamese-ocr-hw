import math
import torch
import torch.nn as nn

class DiacriticsAwareCrossAttention(nn.Module):
    """
    DACA (Diacritics-Aware Cross-Attention) fuses global context features
    with high-resolution local features to reinforce diacritic details during decoding.
    """
    def __init__(self, dim_q, dim_k, dim_out, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        
        # Projections
        self.q_proj = nn.Linear(dim_q, dim_out)
        self.k_proj = nn.Linear(dim_k, dim_out)
        self.v_proj = nn.Linear(dim_k, dim_out)
        
        # PyTorch Multi-head Attention
        self.mha = nn.MultiheadAttention(embed_dim=dim_out, num_heads=num_heads, batch_first=True)
        
        # Layer Normalization and Output Projection
        self.norm = nn.LayerNorm(dim_out)
        self.out_proj = nn.Linear(dim_out, dim_out)
        
        # Residual projection if dim_q differs from dim_out
        if dim_q != dim_out:
            self.res_proj = nn.Linear(dim_q, dim_out)
        else:
            self.res_proj = nn.Identity()
        
    def forward(self, global_feat, local_feat):
        """
        Args:
            global_feat (Tensor): Queries [Batch, Seq_Q, Dim_Q] (context features)
            local_feat (Tensor): Keys/Values [Batch, Seq_K, Dim_K] (high-res local features)
        Returns:
            fused_feat (Tensor): [Batch, Seq_Q, Dim_Out]
        """
        # 1. Project Query, Key, and Value
        q = self.q_proj(global_feat)  # [B, Seq_Q, Dim_Out]
        k = self.k_proj(local_feat)   # [B, Seq_K, Dim_Out]
        v = self.v_proj(local_feat)   # [B, Seq_K, Dim_Out]
        
        # 2. Multi-head Cross Attention
        # Query: global context, Key/Value: local diacritics
        attn_out, attn_weights = self.mha(query=q, key=k, value=v)  # [B, Seq_Q, Dim_Out]
        
        # 3. Residual connection and LayerNorm
        residual = self.res_proj(global_feat)
        fused = self.norm(residual + attn_out)
        
        # 4. Out projection
        fused_feat = self.out_proj(fused)
        
        return fused_feat
