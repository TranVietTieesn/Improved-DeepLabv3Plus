"""
EfficientNet Backbone for Semantic Segmentation.

This module provides a wrapper for EfficientNet models from timm library,
adapted for use with DeepLabv3+ architecture.

EfficientNet features:
- Compound scaling (depth, width, resolution)
- Squeeze-Excitation (SE) attention blocks
- MBConv (Mobile Inverted Bottleneck) blocks
- Efficient for smaller datasets

Supported variants: b0, b1, b2, b3, b4, b5, b6, b7
"""

import torch
import torch.nn as nn
import timm


class EfficientNetBackbone(nn.Module):
    """
    EfficientNet backbone wrapper for semantic segmentation.
    
    EfficientNet uses compound scaling and SE attention blocks,
    making it particularly effective for medical imaging tasks.
    
    Args:
        variant: Model size - 'b0' to 'b7'
        pretrained: Whether to load ImageNet pretrained weights
        in_chans: Number of input channels (default 1 for grayscale, 3 for RGB)
        downsample_factor: Output stride (8 or 16, default 16)
    """
    
    # EfficientNet model configurations
    VARIANTS = {
        'b0': {
            'model_name': 'efficientnet_b0.ra_in1k',
            'dims': [16, 24, 40, 112, 320],  # Stage channels
        },
        'b1': {
            'model_name': 'efficientnet_b1.ra_in1k',
            'dims': [16, 24, 40, 112, 320],
        },
        'b2': {
            'model_name': 'efficientnet_b2.ra_in1k',
            'dims': [16, 24, 48, 120, 352],
        },
        'b3': {
            'model_name': 'efficientnet_b3.ra_in1k',
            'dims': [24, 32, 48, 136, 384],
        },
        'b4': {
            'model_name': 'efficientnet_b4.ra_in1k',
            'dims': [24, 32, 56, 160, 448],
        },
        'b5': {
            'model_name': 'efficientnet_b5.ra_in1k',
            'dims': [24, 40, 64, 176, 512],
        },
        'b6': {
            'model_name': 'efficientnet_b6.ra_in1k',
            'dims': [32, 40, 72, 200, 576],
        },
        'b7': {
            'model_name': 'efficientnet_b7.ra_in1k',
            'dims': [32, 48, 80, 224, 640],
        },
    }
    
    def __init__(self, variant='b7', pretrained=True, in_chans=1, downsample_factor=16):
        super(EfficientNetBackbone, self).__init__()
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Variant '{variant}' not supported. Choose from: {list(self.VARIANTS.keys())}")
        
        self.variant = variant
        self.in_chans = in_chans
        self.config = self.VARIANTS[variant]
        self.downsample_factor = downsample_factor
        
        # Create EfficientNet model with feature extraction
        # EfficientNet has 5 stages, we use indices 1, 2, 3, 4 (skip stage 0)
        self.efficientnet = timm.create_model(
            self.config['model_name'],
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),  # Extract from stages 1-4
            in_chans=in_chans,
        )
        
        # Store output channel info
        # EfficientNet stage outputs: [stage1, stage2, stage3, stage4]
        # Stage 1: 1/4 resolution (low-level)
        # Stage 2: 1/8 resolution
        # Stage 3: 1/16 resolution
        # Stage 4: 1/32 resolution (high-level)
        self.dims = self.config['dims'][1:]  # Skip stage 0, use stages 1-4
        self.out_channels = self.dims[3]  # Stage 4 output channels
        self.low_level_channels = self.dims[0]  # Stage 1 output channels
    
    def forward(self, x):
        """
        Forward pass through EfficientNet backbone.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            low_level_features: Features from stage 1 (B, dims[0], H/4, W/4)
            x: Features from stage 4 (B, dims[3], H/32, W/32)
        """
        # Get multi-scale features from stages 1-4
        features = self.efficientnet(x)
        
        # features[0]: Stage 1 output (1/4 resolution) -> low_level_features
        # features[1]: Stage 2 output (1/8 resolution)
        # features[2]: Stage 3 output (1/16 resolution)
        # features[3]: Stage 4 output (1/32 resolution) -> main features
        
        low_level_features = features[0]  # (B, dims[0], H/4, W/4)
        
        if self.downsample_factor == 8:
            x = features[2]  # Use stage 3 for downsample_factor=8
        else:
            x = features[3]  # Use stage 4 for downsample_factor=16
        
        return low_level_features, x
    
    def get_output_channels(self):
        """Returns (low_level_channels, out_channels) for DeepLab decoder."""
        if self.downsample_factor == 8:
            return self.low_level_channels, self.dims[2]
        return self.low_level_channels, self.out_channels
    
    def get_all_features(self, x):
        """
        Get features from all 4 stages.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            List of features from stages 1-4
        """
        return self.efficientnet(x)


def efficientnet_backbone(variant='b7', pretrained=True, in_chans=1, downsample_factor=16):
    """
    Factory function to create EfficientNet backbone.
    
    Args:
        variant: 'b0' to 'b7'
        pretrained: Whether to load ImageNet pretrained weights
        in_chans: Number of input channels (1 for grayscale, 3 for RGB)
        downsample_factor: Output stride (8 or 16)
        
    Returns:
        EfficientNetBackbone module
        
    Example:
        >>> backbone = efficientnet_backbone('b7', pretrained=True, in_chans=1)
        >>> x = torch.randn(1, 1, 448, 448)
        >>> low_feat, high_feat = backbone(x)
        >>> print(low_feat.shape, high_feat.shape)
        torch.Size([1, 48, 112, 112]) torch.Size([1, 640, 14, 14])
    """
    return EfficientNetBackbone(
        variant=variant,
        pretrained=pretrained,
        in_chans=in_chans,
        downsample_factor=downsample_factor
    )
