"""
Swin Transformer Backbone for DeepLabv3+
=========================================
Wrapper around timm's Swin Transformer implementation.
Supports Tiny/Small/Base variants with pretrained ImageNet weights.
Handles 1-channel input by replicating to 3 channels.
"""

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("Please install timm: pip install timm")


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
    """
    
    # Swin model configurations
    VARIANTS = {
        'tiny': {
            'model_name': 'swin_tiny_patch4_window7_224',
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'out_channels': 768,
            'low_level_channels': 96,
        },
        'small': {
            'model_name': 'swin_small_patch4_window7_224',
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'out_channels': 768,
            'low_level_channels': 96,
        },
        'base': {
            'model_name': 'swin_base_patch4_window7_224',
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'out_channels': 1024,
            'low_level_channels': 128,
        },
    }
    
    def __init__(self, variant='tiny', pretrained=True, in_chans=1, downsample_factor=16):
        super(SwinBackbone, self).__init__()
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Unsupported variant '{variant}'. Choose from {list(self.VARIANTS.keys())}")
        
        self.variant = variant
        self.in_chans = in_chans
        self.config = self.VARIANTS[variant]
        self.downsample_factor = downsample_factor
        
        # Create Swin model with feature extraction
        self.swin = timm.create_model(
            self.config['model_name'],
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Extract from all 4 stages
        )
        
        # Handle 1-channel input
        if in_chans == 1:
            # Get original conv weights
            orig_conv = self.swin.patch_embed.proj
            # Create new conv for 1 channel
            self.input_conv = nn.Conv2d(
                1, orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None
            )
            # Initialize with mean of RGB weights
            if pretrained:
                with torch.no_grad():
                    self.input_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
                    if orig_conv.bias is not None:
                        self.input_conv.bias.copy_(orig_conv.bias)
            # Replace the patch embedding projection
            self.swin.patch_embed.proj = self.input_conv
        
        # Store output channel info for DeepLab
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
        # Get multi-scale features from all stages
        features = self.swin(x)
        
        # features[0]: Stage 1 output (1/4 resolution) -> low_level_features
        # features[1]: Stage 2 output (1/8 resolution)
        # features[2]: Stage 3 output (1/16 resolution)
        # features[3]: Stage 4 output (1/32 resolution) -> main features
        
        low_level_features = features[0]  # (B, 96/128, H/4, W/4)
        
        if self.downsample_factor == 8:
            # Use stage 3 output for downsample_factor=8
            x = features[2]  # (B, C, H/16, W/16)
        else:
            # Use stage 4 output for downsample_factor=16
            x = features[3]  # (B, 768/1024, H/32, W/32)
        
        return low_level_features, x
    
    def get_output_channels(self):
        """Returns (low_level_channels, out_channels) for DeepLab decoder."""
        if self.downsample_factor == 8:
            # Stage 3 channels
            return self.low_level_channels, self.config['embed_dim'] * 4
        return self.low_level_channels, self.out_channels


def swin_transformer(variant='tiny', pretrained=True, in_chans=1, downsample_factor=16):
    """
    Factory function to create Swin Transformer backbone.
    
    Args:
        variant: 'tiny', 'small', or 'base'
        pretrained: Whether to load ImageNet pretrained weights
        in_chans: Number of input channels (1 for grayscale, 3 for RGB)
        downsample_factor: Output stride (8 or 16)
        
    Returns:
        SwinBackbone module
        
    Example:
        >>> backbone = swin_transformer('tiny', pretrained=True, in_chans=1)
        >>> x = torch.randn(1, 1, 512, 512)
        >>> low_feat, high_feat = backbone(x)
        >>> print(low_feat.shape, high_feat.shape)
    """
    return SwinBackbone(
        variant=variant,
        pretrained=pretrained,
        in_chans=in_chans,
        downsample_factor=downsample_factor
    )
