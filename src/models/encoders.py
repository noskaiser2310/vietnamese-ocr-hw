import torch
import torch.nn as nn
import torchvision.models as models

class GlobalContextEncoder(nn.Module):
    """
    Extracts global context features from the text line image.
    Uses downsampled convolutional features from a ResNet backbone and
    projects them into a sequence of features along the horizontal axis.
    """
    def __init__(self, backbone_name="resnet34", pretrained=True, global_dim=768):
        super().__init__()
        # Load ResNet backbone
        if backbone_name == "resnet34":
            if pretrained:
                weights = models.ResNet34_Weights.DEFAULT
            else:
                weights = None
            self.backbone = models.resnet34(weights=weights)
            out_channels = 512
        elif backbone_name == "resnet50":
            if pretrained:
                weights = models.ResNet50_Weights.DEFAULT
            else:
                weights = None
            self.backbone = models.resnet50(weights=weights)
            out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Remove average pooling and fc layer
        self.conv_layers = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,  # [B, C, H/4, W/4]
            self.backbone.layer2,  # [B, C, H/8, W/8]
            self.backbone.layer3,  # [B, C, H/16, W/16]
            self.backbone.layer4   # [B, C, H/32, W/32]
        )
        
        # Adaptive pooling to collapse vertical axis (Height -> 1)
        self.vertical_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Projection layer to hidden dim
        self.proj = nn.Linear(out_channels, global_dim)
        
    def forward(self, x):
        # x shape: [Batch, 3, Height (128), Width]
        features = self.conv_layers(x)  # shape: [B, C, H/32 (4), W/32]
        
        # Collapse height dimension
        pooled = self.vertical_pool(features)  # shape: [B, C, 1, W/32]
        pooled = pooled.squeeze(2)             # shape: [B, C, W/32]
        pooled = pooled.permute(0, 2, 1)       # shape: [B, W/32, C]
        
        # Project to global hidden dim
        global_feat = self.proj(pooled)        # shape: [B, W/32, global_dim]
        return global_feat


class SpatialAttention(nn.Module):
    """
    Spatial Attention module to weigh local regions focusing on diacritics.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        attn_map = self.sigmoid(self.conv(x))
        return x * attn_map, attn_map


class LocalDiacriticsDetector(nn.Module):
    """
    High-Resolution local feature extractor based on Feature Pyramid Network (FPN).
    Preserves high-resolution details of small diacritics using Spatial Attention.
    """
    def __init__(self, backbone_name="resnet34", pretrained=True, feature_dim=256):
        super().__init__()
        # Load a separate lightweight backbone to avoid shared state complexity
        if backbone_name in ["resnet34", "resnet50"]:
            if pretrained:
                weights = models.ResNet34_Weights.DEFAULT if backbone_name == "resnet34" else models.ResNet50_Weights.DEFAULT
            else:
                weights = None
            backbone = models.resnet34(weights=weights) if backbone_name == "resnet34" else models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Extract low-level feature layers
        self.init_conv = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )  # [B, 64, H/4 (32), W/4]
        
        self.layer1 = backbone.layer1  # [B, 64, H/4 (32), W/4]
        self.layer2 = backbone.layer2  # [B, 128, H/8 (16), W/8]
        
        # Lateral connections (FPN)
        self.lat_layer2 = nn.Conv2d(128, feature_dim, kernel_size=1)
        self.lat_layer1 = nn.Conv2d(64, feature_dim, kernel_size=1)
        
        # Anti-aliasing convolution
        self.smooth_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        
        # Spatial Attention to isolate tiny tone marks and diacritics
        self.spatial_attention = SpatialAttention(feature_dim)
        
        # Downsample spatial feature slightly to save memory in DACA
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)  # [B, feature_dim, H/8 (16), W/8]
        
    def forward(self, x):
        # x shape: [Batch, 3, Height (128), Width]
        
        # Bottom-up pathway
        c1 = self.init_conv(x)  # shape: [B, 64, 32, W_c1]
        c2 = self.layer1(c1)    # shape: [B, 64, 32, W_c2]
        c3 = self.layer2(c2)    # shape: [B, 128, 16, W_c3]
        
        # Top-down pathway (FPN Fusion)
        p3 = self.lat_layer2(c3)  # shape: [B, feature_dim, 16, W_c3]
        p2 = self.lat_layer1(c2)  # shape: [B, feature_dim, 32, W_c2]
        
        # Upsample p3 to match shape of p2
        up_p3 = nn.functional.interpolate(p3, size=(p2.shape[2], p2.shape[3]), mode="nearest")
        
        # Add lateral features
        fused = p2 + up_p3
        fused = self.smooth_conv(fused)  # shape: [B, feature_dim, 32, W_c2]
        
        # Apply Spatial Attention
        attended, attn_map = self.spatial_attention(fused)
        
        # Max pool to reduce sequence length for Transformer Cross-Attention
        pooled = self.downsample(attended)  # shape: [B, feature_dim, 16, W/8]
        
        # Flatten spatial dimensions H and W into sequence dimension
        b, c, h, w = pooled.shape
        local_feat = pooled.permute(0, 2, 3, 1).reshape(b, h * w, c)  # shape: [B, H*W, feature_dim]
        
        return local_feat, attn_map
