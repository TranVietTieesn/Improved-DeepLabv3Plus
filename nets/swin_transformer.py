"""
Swin Transformer Backbone for DeepLabv3+
=========================================
Wrapper around timm's Swin Transformer implementation.
Supports Tiny/Small/Base variants with pretrained ImageNet weights.
Handles 1-channel input by replicating to 3 channels.

IMPORTANT: Image size must be divisible by 28 (patch_size=4 × window_size=7)
Valid sizes: 224, 252, 280, 308, 336, 364, 392, 420, 448, 476, 504, 560, 672, 896...
If you need 512, use 504 or 560 instead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


def get_valid_swin_sizes(min_size=224, max_size=1024):
    """
    Get valid image sizes for Swin Transformer (must be divisible by 28).
    patch_size=4, window_size=7, so 4*7=28
    """
    return [28 * n for n in range(min_size // 28, max_size // 28 + 1)]


class SwinBackbone(nn.Module):
    """
    Swin Transformer backbone wrapper for semantic segmentation.
    
    Extracts multi-scale features suitable for DeepLabv3+ decoder:
    - low_level_features: from stage 1 (1/4 resolution)
    - high_level_features: from stage 4 (1/32 resolution)
    
    Args:
        variant: 'tiny', 'small', or 'base'
        pretrained: Whether to load ImageNet pretrained weights
        in_chans: Number of input channels (default 1, will be replicated to 3)
        downsample_factor: Output stride (8 or 16, default 16)
        img_size: Input image size (must be divisible by 28). Default 224.
                  Valid sizes: 224, 252, 280, ..., 448, 504, 560, ...
    """
    
    # Swin model configurations
    VARIANTS = {
        'tiny': {
            'model_name': 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'out_channels': 768,
            'low_level_channels': 96,
            'window_size': 7,
            'patch_size': 4,
        },
        'small': {
            'model_name': 'swin_small_patch4_window7_224.ms_in22k_ft_in1k',
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'out_channels': 768,
            'low_level_channels': 96,
            'window_size': 7,
            'patch_size': 4,
        },
        'base': {
            'model_name': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k',
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'out_channels': 1024,
            'low_level_channels': 128,
            'window_size': 7,
            'patch_size': 4,
        },
    }
    
    def __init__(self, variant='tiny', pretrained=True, in_chans=1, 
                 downsample_factor=16, img_size=224):
        super(SwinBackbone, self).__init__()
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant '{variant}'. Choose from {list(self.VARIANTS.keys())}")
        
        self.variant = variant
        self.in_chans = in_chans
        self.config = self.VARIANTS[variant]
        self.downsample_factor = downsample_factor
        self.img_size = img_size
        
        # Validate image size
        window_size = self.config['window_size']
        patch_size = self.config['patch_size']
        required_divisor = window_size * patch_size  # 28 for Swin with window=7, patch=4
        
        if img_size % required_divisor != 0:
            valid_sizes = get_valid_swin_sizes(224, 1024)
            # Find closest valid size
            closest = min(valid_sizes, key=lambda x: abs(x - img_size))
            raise ValueError(
                f"Image size {img_size} is not compatible with Swin Transformer.\n"
                f"Size must be divisible by {required_divisor} (patch_size={patch_size} × window_size={window_size}).\n"
                f"Closest valid size: {closest}\n"
                f"Some valid sizes: {[s for s in valid_sizes if 200 <= s <= 600]}"
            )
        
        # Create Swin model with the specified image size
        self.swin = timm.create_model(
            self.config['model_name'],
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,  # Pass the desired image size
        )
        
        # Handle 1-channel input
        if in_chans == 1:
            orig_conv = self.swin.patch_embed.proj
            self.input_conv = nn.Conv2d(
                1, orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None
            )
            if pretrained:
                with torch.no_grad():
                    self.input_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
                    if orig_conv.bias is not None:
                        self.input_conv.bias.copy_(orig_conv.bias)
            self.swin.patch_embed.proj = self.input_conv
        
        # Store output channel info
        self.out_channels = self.config['out_channels']
        self.low_level_channels = self.config['low_level_channels']
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W) where C=1 or 3
            
        Returns:
            low_level_features: Features from stage 1 (B, 96/128, H/4, W/4)
            x: Features from stage 4 (B, 768/1024, H/32, W/32)
        """
        # Get multi-scale features
        features = self.swin(x)
        
        # Convert from NHWC to NCHW format
        low_level_features = features[0].permute(0, 3, 1, 2).contiguous()
        
        if self.downsample_factor == 8:
            x = features[2].permute(0, 3, 1, 2).contiguous()
        else:
            x = features[3].permute(0, 3, 1, 2).contiguous()
        
        return low_level_features, x
    
    def get_output_channels(self):
        """Returns (low_level_channels, out_channels) for DeepLab decoder."""
        if self.downsample_factor == 8:
            return self.low_level_channels, self.config['embed_dim'] * 4
        return self.low_level_channels, self.out_channels


def swin_transformer(variant='tiny', pretrained=True, in_chans=1, 
                     downsample_factor=16, img_size=224):
    """
    Factory function to create Swin Transformer backbone.
    
    Args:
        variant: 'tiny', 'small', or 'base'
        pretrained: Whether to load ImageNet pretrained weights
        in_chans: Number of input channels (1 for grayscale, 3 for RGB)
        downsample_factor: Output stride (8 or 16)
        img_size: Input image size (must be divisible by 28)
                  Valid sizes: 224, 252, 280, 448, 504, 560, 672...
        
    Returns:
        SwinBackbone module
        
    Example:
        >>> backbone = swin_transformer('tiny', pretrained=True, in_chans=1, img_size=448)
        >>> x = torch.randn(1, 1, 448, 448)
        >>> low_feat, high_feat = backbone(x)
        >>> print(low_feat.shape, high_feat.shape)
    """
    return SwinBackbone(
        variant=variant,
        pretrained=pretrained,
        in_chans=in_chans,
        downsample_factor=downsample_factor,
        img_size=img_size
    )
