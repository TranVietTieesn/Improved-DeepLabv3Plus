"""
DeepLabv3+ with SAM-style Prompt Attention.

This module combines:
- EfficientNet backbone (proven effective for medical imaging)
- ASPP for multi-scale context (from DeepLab)
- Learnable Prompt Tokens with Cross-Attention (inspired by SAM)
- DeepLab-style decoder with skip connections

Key innovation:
- Prompt tokens learn to "query" image features for tumor regions
- Lightweight compared to full SAM
- Combines best of CNN (EfficientNet) and attention mechanisms

Author: Generated for brain tumor segmentation research
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .efficientnet_backbone import EfficientNetBackbone
from .attention import se_block, cbam_block, eca_block


class PromptAttention(nn.Module):
    """
    SAM-style Prompt Attention module (Lightweight version).
    
    Uses fixed internal dimension (256) for efficiency, regardless of input dim.
    """
    
    def __init__(self, dim, num_prompts=8, num_heads=8, dropout=0.1):
        super(PromptAttention, self).__init__()
        
        self.dim = dim
        self.num_prompts = num_prompts
        
        # Fixed internal dimension for efficiency
        self.internal_dim = 256
        self.num_heads = min(num_heads, self.internal_dim // 32)  # Ensure valid heads
        self.head_dim = self.internal_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        # Project input to internal dimension
        self.input_proj = nn.Sequential(
            nn.Conv2d(dim, self.internal_dim, 1, bias=False),
            nn.BatchNorm2d(self.internal_dim),
            nn.ReLU(inplace=True)
        )
        
        # Learnable prompt tokens
        self.prompt_tokens = nn.Parameter(torch.randn(num_prompts, self.internal_dim))
        nn.init.trunc_normal_(self.prompt_tokens, std=0.02)
        
        # Cross-attention projections
        self.q_proj = nn.Linear(self.internal_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.internal_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.internal_dim, self.internal_dim)
        
        # Lightweight spatial projection
        self.spatial_proj = nn.Sequential(
            nn.Linear(self.internal_dim, self.internal_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.internal_dim, dim)  # Project back to original dim
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """Apply prompt attention to image features."""
        B, C, H, W = x.shape
        
        # Project to internal dimension
        x_proj = self.input_proj(x)  # (B, internal_dim, H, W)
        
        # Reshape to sequence
        x_seq = x_proj.flatten(2).transpose(1, 2)  # (B, H*W, internal_dim)
        
        # Get prompts
        prompts = self.prompt_tokens.unsqueeze(0).expand(B, -1, -1)  # (B, num_prompts, internal_dim)
        
        # Cross-attention: prompts query image features
        Q = self.q_proj(prompts)  # (B, num_prompts, internal_dim)
        K = self.k_proj(x_seq)    # (B, H*W, internal_dim)
        V = self.v_proj(x_seq)    # (B, H*W, internal_dim)
        
        # Multi-head attention
        Q = Q.view(B, self.num_prompts, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, H*W, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        attended = attn @ V  # (B, heads, num_prompts, head_dim)
        attended = attended.transpose(1, 2).reshape(B, self.num_prompts, self.internal_dim)
        
        # Aggregate prompts: mean pooling
        prompt_agg = attended.mean(dim=1)  # (B, internal_dim)
        
        # Project to channel attention weights
        channel_weight = self.spatial_proj(prompt_agg)  # (B, C)
        channel_weight = channel_weight.view(B, C, 1, 1).sigmoid()
        
        # Apply as channel attention
        x_enhanced = x * channel_weight + x
        
        return x_enhanced


class PromptASPP(nn.Module):
    """
    ASPP module enhanced with Prompt Attention.
    
    Combines multi-scale context (ASPP) with learned prompts (SAM-style).
    """
    
    def __init__(self, dim_in, dim_out, num_prompts=8, rate=1, bn_mom=0.1):
        super(PromptASPP, self).__init__()
        
        # Standard ASPP branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling branch
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        
        # Prompt Attention - applied after ASPP concat
        self.prompt_attn = PromptAttention(
            dim=dim_out * 5,
            num_prompts=num_prompts,
            num_heads=8
        )
        
        # Final projection
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        b, c, row, col = x.size()
        
        # Multi-scale features from ASPP
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        
        # Global context
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        
        # Concatenate all branches
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        
        # Apply Prompt Attention - key innovation!
        feature_enhanced = self.prompt_attn(feature_cat)
        
        # Final projection
        result = self.conv_cat(feature_enhanced)
        
        return result


class DeepLabPrompt(nn.Module):
    """
    DeepLabv3+ with SAM-style Prompt Attention.
    
    Combines EfficientNet backbone with prompt-enhanced ASPP decoder.
    
    Args:
        num_classes: Number of output classes
        efficientnet_variant: EfficientNet variant ('b0' to 'b7')
        num_prompts: Number of learnable prompt tokens
        decoder_channels: Decoder channel size
        pretrained: Whether to load pretrained weights
        in_chans: Number of input channels (1 for grayscale)
        downsample_factor: Output stride (8 or 16)
    """
    
    def __init__(
        self,
        num_classes,
        efficientnet_variant='b7',
        num_prompts=8,
        decoder_channels=256,
        pretrained=True,
        in_chans=1,
        downsample_factor=16
    ):
        super(DeepLabPrompt, self).__init__()
        
        self.num_classes = num_classes
        self.efficientnet_variant = efficientnet_variant
        self.num_prompts = num_prompts
        self.decoder_channels = decoder_channels
        self.downsample_factor = downsample_factor
        
        # EfficientNet backbone
        self.backbone = EfficientNetBackbone(
            variant=efficientnet_variant,
            pretrained=pretrained,
            in_chans=in_chans,
            downsample_factor=downsample_factor
        )
        
        # Get backbone output channels
        low_level_ch, high_level_ch = self.backbone.get_output_channels()
        
        # Prompt-enhanced ASPP
        self.aspp = PromptASPP(
            dim_in=high_level_ch,
            dim_out=decoder_channels,
            num_prompts=num_prompts,
            rate=16 // downsample_factor
        )
        
        # Calculate shortcut channels
        shortcut_ch = max(48, decoder_channels // 4)
        
        # Low-level features processing
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_ch, shortcut_ch, 1),
            nn.BatchNorm2d(shortcut_ch),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.cat_conv = nn.Sequential(
            nn.Conv2d(shortcut_ch + decoder_channels, decoder_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Conv2d(decoder_channels, decoder_channels, 3, stride=1, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # Final classification
        self.cls_conv = nn.Conv2d(decoder_channels, num_classes, 1, stride=1)
    
    def forward(self, x):
        """Forward pass."""
        H, W = x.size(2), x.size(3)
        
        # Backbone: get low-level and high-level features
        low_level_features, high_level_features = self.backbone(x)
        
        # Prompt-enhanced ASPP
        x = self.aspp(high_level_features)
        
        # Process low-level features
        low_level_features = self.shortcut_conv(low_level_features)
        
        # Upsample and concatenate
        x = F.interpolate(
            x,
            size=(low_level_features.size(2), low_level_features.size(3)),
            mode='bilinear',
            align_corners=True
        )
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        
        # Final classification and upsample
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_flops(self, input_size=(1, 1, 512, 512), verbose=False):
        """Count FLOPs (requires thop library)."""
        try:
            from thop import profile
        except ImportError:
            raise ImportError("Install thop: pip install thop")
        
        import copy
        device = next(self.parameters()).device
        model_copy = copy.deepcopy(self)
        model_copy.eval()
        
        dummy_input = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            flops, params = profile(model_copy, inputs=(dummy_input,), verbose=verbose)
        
        del model_copy
        return flops / 1e9, params
    
    def get_model_info(self, input_size=(1, 1, 512, 512)):
        """Get comprehensive model information."""
        info = {
            'model': 'DeepLabPrompt (SAM-style)',
            'efficientnet_variant': self.efficientnet_variant,
            'num_prompts': self.num_prompts,
            'decoder_channels': self.decoder_channels,
            'num_classes': self.num_classes,
            'parameters': self.count_parameters(),
            'parameters_M': self.count_parameters() / 1e6,
        }
        
        try:
            gflops, _ = self.count_flops(input_size=input_size)
            info['gflops'] = gflops
        except ImportError:
            info['gflops'] = 'N/A (install thop)'
        
        return info


def deeplabv3_plus_prompt(
    num_classes,
    efficientnet_variant='b7',
    num_prompts=8,
    decoder_channels=256,
    pretrained=True,
    in_chans=1,
    downsample_factor=16
):
    """
    Factory function to create DeepLabv3+ with Prompt Attention.
    
    Args:
        num_classes: Number of output classes
        efficientnet_variant: 'b0' to 'b7'
        num_prompts: Number of learnable prompt tokens (default 8)
        decoder_channels: Decoder size (256 default, 512 or 1024 for more capacity)
        
    Example:
        >>> model = deeplabv3_plus_prompt(num_classes=4, num_prompts=8)
        >>> x = torch.randn(1, 1, 512, 512)
        >>> output = model(x)
    """
    return DeepLabPrompt(
        num_classes=num_classes,
        efficientnet_variant=efficientnet_variant,
        num_prompts=num_prompts,
        decoder_channels=decoder_channels,
        pretrained=pretrained,
        in_chans=in_chans,
        downsample_factor=downsample_factor
    )


# Recommended configurations
CONFIGS = {
    'light': {
        'efficientnet_variant': 'b4',
        'num_prompts': 4,
        'decoder_channels': 256,
    },
    'medium': {
        'efficientnet_variant': 'b5',
        'num_prompts': 8,
        'decoder_channels': 256,
    },
    'heavy': {
        'efficientnet_variant': 'b7',
        'num_prompts': 8,
        'decoder_channels': 512,
    },
}
