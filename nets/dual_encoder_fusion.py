"""
Dual Encoder Fusion Modules.

This module provides fusion mechanisms to combine features from two encoder branches
(e.g., ConvNeXt + PVTv2) for semantic segmentation.

Components:
- ChannelAlignConv: Align channel dimensions between two feature maps
- GatedFusion: Single-level gated fusion with learnable weights
- MultiLevelGatedFusion: Multi-level fusion at low and high levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAlignConv(nn.Module):
    """
    1x1 convolution to align channel dimensions.
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
    """
    
    def __init__(self, in_channels, out_channels):
        super(ChannelAlignConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class GatedFusion(nn.Module):
    """
    Gated Fusion module to combine features from two branches.
    
    Uses a learnable gate to adaptively blend features:
        gate = sigmoid(conv(concat(feat1, feat2)))
        output = gate * feat1_aligned + (1 - gate) * feat2_aligned
    
    Args:
        in_channels_1: Channels from first branch
        in_channels_2: Channels from second branch
        out_channels: Output channels after fusion
    """
    
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super(GatedFusion, self).__init__()
        
        self.in_channels_1 = in_channels_1
        self.in_channels_2 = in_channels_2
        self.out_channels = out_channels
        
        # Align channels from both branches to output channels
        self.align_1 = ChannelAlignConv(in_channels_1, out_channels)
        self.align_2 = ChannelAlignConv(in_channels_2, out_channels)
        
        # Gate network: learns which branch to emphasize
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, feat1, feat2):
        """
        Fuse features from two branches.
        
        Args:
            feat1: Features from branch 1 (e.g., ConvNeXt)
            feat2: Features from branch 2 (e.g., PVTv2)
            
        Returns:
            Fused features with out_channels
        """
        # Align spatial dimensions if needed
        if feat1.shape[2:] != feat2.shape[2:]:
            feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=True)
        
        # Align channel dimensions
        feat1_aligned = self.align_1(feat1)
        feat2_aligned = self.align_2(feat2)
        
        # Compute gate
        concat = torch.cat([feat1_aligned, feat2_aligned], dim=1)
        gate = self.gate_conv(concat)
        
        # Blend features
        fused = gate * feat1_aligned + (1 - gate) * feat2_aligned
        
        return fused


class MultiLevelGatedFusion(nn.Module):
    """
    Multi-level Gated Fusion for Dual Encoder.
    
    Fuses features at two levels:
    - Low-level: Stage 1 features (1/4 resolution) for decoder
    - High-level: Stage 4 features (1/32 resolution) for ASPP
    
    Args:
        convnext_dims: Channel dimensions for ConvNeXt [stage1, stage2, stage3, stage4]
        pvtv2_dims: Channel dimensions for PVTv2 [stage1, stage2, stage3, stage4]
        low_level_out: Output channels for low-level fusion
        high_level_out: Output channels for high-level fusion
    """
    
    def __init__(self, convnext_dims, pvtv2_dims, low_level_out=128, high_level_out=512):
        super(MultiLevelGatedFusion, self).__init__()
        
        self.convnext_dims = convnext_dims
        self.pvtv2_dims = pvtv2_dims
        
        # Low-level fusion (Stage 1)
        self.low_level_fusion = GatedFusion(
            in_channels_1=convnext_dims[0],  # e.g., 96 for ConvNeXt-Tiny
            in_channels_2=pvtv2_dims[0],     # e.g., 64 for PVTv2-B2
            out_channels=low_level_out
        )
        
        # High-level fusion (Stage 4)
        self.high_level_fusion = GatedFusion(
            in_channels_1=convnext_dims[3],  # e.g., 768 for ConvNeXt-Tiny
            in_channels_2=pvtv2_dims[3],     # e.g., 512 for PVTv2-B2
            out_channels=high_level_out
        )
        
        self.low_level_channels = low_level_out
        self.high_level_channels = high_level_out
    
    def forward(self, convnext_features, pvtv2_features):
        """
        Fuse features from both backbones at multiple levels.
        
        Args:
            convnext_features: List of 4 feature maps from ConvNeXt
            pvtv2_features: List of 4 feature maps from PVTv2
            
        Returns:
            low_level_fused: Fused low-level features for decoder
            high_level_fused: Fused high-level features for ASPP
        """
        # Low-level fusion (Stage 1)
        low_level_fused = self.low_level_fusion(
            convnext_features[0],
            pvtv2_features[0]
        )
        
        # High-level fusion (Stage 4)
        high_level_fused = self.high_level_fusion(
            convnext_features[3],
            pvtv2_features[3]
        )
        
        return low_level_fused, high_level_fused
    
    def get_output_channels(self):
        """Returns (low_level_channels, high_level_channels)."""
        return self.low_level_channels, self.high_level_channels


class DualEncoderWithFusion(nn.Module):
    """
    Complete Dual Encoder module with Multi-level Gated Fusion.
    
    Combines ConvNeXt and PVTv2 backbones with fusion for DeepLabv3+.
    
    Args:
        convnext_backbone: ConvNeXtBackbone module
        pvtv2_backbone: PVTv2Backbone module
        low_level_out: Output channels for low-level fusion
        high_level_out: Output channels for high-level fusion
    """
    
    def __init__(self, convnext_backbone, pvtv2_backbone, low_level_out=128, high_level_out=512):
        super(DualEncoderWithFusion, self).__init__()
        
        self.convnext = convnext_backbone
        self.pvtv2 = pvtv2_backbone
        
        # Create fusion module
        self.fusion = MultiLevelGatedFusion(
            convnext_dims=convnext_backbone.dims,
            pvtv2_dims=pvtv2_backbone.dims,
            low_level_out=low_level_out,
            high_level_out=high_level_out
        )
        
        self.low_level_channels = low_level_out
        self.out_channels = high_level_out
    
    def forward(self, x):
        """
        Forward pass through both backbones with fusion.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            low_level_features: Fused low-level features for decoder
            high_level_features: Fused high-level features for ASPP
        """
        # Get features from both backbones
        convnext_feats = self.convnext.get_all_features(x)
        pvtv2_feats = self.pvtv2.get_all_features(x)
        
        # Fuse features at multiple levels
        low_level_fused, high_level_fused = self.fusion(convnext_feats, pvtv2_feats)
        
        return low_level_fused, high_level_fused
    
    def get_output_channels(self):
        """Returns (low_level_channels, out_channels) for DeepLab decoder."""
        return self.low_level_channels, self.out_channels
