"""
ConvNeXt Backbone for Semantic Segmentation.

This module provides a wrapper for ConvNeXt models from timm library,
adapted for use with DeepLabv3+ architecture.

Supported variants: tiny, small, base, large
"""

import torch
import torch.nn as nn
import timm


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone wrapper for semantic segmentation.
    
    ConvNeXt is a pure convolutional model that incorporates design choices
    from Vision Transformers, achieving strong performance with efficiency.
    
    Args:
        variant: Model size - 'tiny', 'small', 'base', or 'large'
        pretrained: Whether to load ImageNet pretrained weights
        in_chans: Number of input channels (default 1 for grayscale, 3 for RGB)
        downsample_factor: Output stride (8 or 16, default 16)
    """
    
    # ConvNeXt model configurations
    # Using base model names for compatibility with different timm versions
    VARIANTS = {
        'tiny': {
            'model_name': 'convnext_tiny',
            'dims': [96, 192, 384, 768],
        },
        'small': {
            'model_name': 'convnext_small',
            'dims': [96, 192, 384, 768],
        },
        'base': {
            'model_name': 'convnext_base',
            'dims': [128, 256, 512, 1024],
        },
        'large': {
            'model_name': 'convnext_large',
            'dims': [192, 384, 768, 1536],
        },
    }
    
    def __init__(self, variant='tiny', pretrained=True, in_chans=1, downsample_factor=16):
        super(ConvNeXtBackbone, self).__init__()
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Variant '{variant}' not supported. Choose from: {list(self.VARIANTS.keys())}")
        
        self.variant = variant
        self.in_chans = in_chans
        self.config = self.VARIANTS[variant]
        self.downsample_factor = downsample_factor
        
        # Create ConvNeXt model with feature extraction
        self.convnext = timm.create_model(
            self.config['model_name'],
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),  # Extract from all 4 stages
            in_chans=in_chans,  # timm handles input channel conversion
        )
        
        # Store output channel info for DeepLab
        self.dims = self.config['dims']
        self.out_channels = self.dims[3]  # Stage 4 output channels
        self.low_level_channels = self.dims[0]  # Stage 1 output channels
    
    def forward(self, x):
        """
        Forward pass through ConvNeXt backbone.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            low_level_features: Features from stage 1 (B, dims[0], H/4, W/4)
            x: Features from stage 4 (B, dims[3], H/32, W/32)
        """
        # Get multi-scale features from all stages
        features = self.convnext(x)
        
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
            List of features from all 4 stages
        """
        return self.convnext(x)


def convnext_backbone(variant='tiny', pretrained=True, in_chans=1, downsample_factor=16):
    """
    Factory function to create ConvNeXt backbone.
    
    Args:
        variant: 'tiny', 'small', 'base', or 'large'
        pretrained: Whether to load ImageNet pretrained weights
        in_chans: Number of input channels (1 for grayscale, 3 for RGB)
        downsample_factor: Output stride (8 or 16)
        
    Returns:
        ConvNeXtBackbone module
        
    Example:
        >>> backbone = convnext_backbone('tiny', pretrained=True, in_chans=1)
        >>> x = torch.randn(1, 1, 448, 448)
        >>> low_feat, high_feat = backbone(x)
        >>> print(low_feat.shape, high_feat.shape)
        torch.Size([1, 96, 112, 112]) torch.Size([1, 768, 14, 14])
    """
    return ConvNeXtBackbone(
        variant=variant,
        pretrained=pretrained,
        in_chans=in_chans,
        downsample_factor=downsample_factor
    )
