"""
PVTv2 (Pyramid Vision Transformer v2) Backbone for Semantic Segmentation.

This module provides a wrapper for PVTv2 models from timm library,
adapted for use with DeepLabv3+ architecture.

PVTv2 features:
- Linear Spatial Reduction Attention (efficient O(n) complexity)
- Overlapping Patch Embedding
- Convolutional Feed-Forward Network

Supported variants: b0, b1, b2, b3, b4, b5
"""

import torch
import torch.nn as nn
import timm


class PVTv2Backbone(nn.Module):
    """
    PVTv2 backbone wrapper for semantic segmentation.
    
    PVTv2 is an efficient vision transformer with linear complexity attention,
    making it suitable for dense prediction tasks.
    
    Args:
        variant: Model size - 'b0', 'b1', 'b2', 'b3', 'b4', or 'b5'
        pretrained: Whether to load ImageNet pretrained weights
        in_chans: Number of input channels (default 1 for grayscale, 3 for RGB)
        downsample_factor: Output stride (8 or 16, default 16)
    """
    
    # PVTv2 model configurations
    VARIANTS = {
        'b0': {
            'model_name': 'pvt_v2_b0',
            'dims': [32, 64, 160, 256],
        },
        'b1': {
            'model_name': 'pvt_v2_b1',
            'dims': [64, 128, 320, 512],
        },
        'b2': {
            'model_name': 'pvt_v2_b2',
            'dims': [64, 128, 320, 512],
        },
        'b3': {
            'model_name': 'pvt_v2_b3',
            'dims': [64, 128, 320, 512],
        },
        'b4': {
            'model_name': 'pvt_v2_b4',
            'dims': [64, 128, 320, 512],
        },
        'b5': {
            'model_name': 'pvt_v2_b5',
            'dims': [64, 128, 320, 512],
        },
    }
    
    def __init__(self, variant='b2', pretrained=True, in_chans=1, downsample_factor=16):
        super(PVTv2Backbone, self).__init__()
        
        if variant not in self.VARIANTS:
            raise ValueError(f"Variant '{variant}' not supported. Choose from: {list(self.VARIANTS.keys())}")
        
        self.variant = variant
        self.in_chans = in_chans
        self.config = self.VARIANTS[variant]
        self.downsample_factor = downsample_factor
        
        # Create PVTv2 model with feature extraction
        self.pvt = timm.create_model(
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
        Forward pass through PVTv2 backbone.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            low_level_features: Features from stage 1 (B, dims[0], H/4, W/4)
            x: Features from stage 4 (B, dims[3], H/32, W/32)
        """
        # Get multi-scale features from all stages
        features = self.pvt(x)
        
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
        return self.pvt(x)


def pvtv2_backbone(variant='b2', pretrained=True, in_chans=1, downsample_factor=16):
    """
    Factory function to create PVTv2 backbone.
    
    Args:
        variant: 'b0', 'b1', 'b2', 'b3', 'b4', or 'b5'
        pretrained: Whether to load ImageNet pretrained weights
        in_chans: Number of input channels (1 for grayscale, 3 for RGB)
        downsample_factor: Output stride (8 or 16)
        
    Returns:
        PVTv2Backbone module
        
    Example:
        >>> backbone = pvtv2_backbone('b2', pretrained=True, in_chans=1)
        >>> x = torch.randn(1, 1, 448, 448)
        >>> low_feat, high_feat = backbone(x)
        >>> print(low_feat.shape, high_feat.shape)
        torch.Size([1, 64, 112, 112]) torch.Size([1, 512, 14, 14])
    """
    return PVTv2Backbone(
        variant=variant,
        pretrained=pretrained,
        in_chans=in_chans,
        downsample_factor=downsample_factor
    )
