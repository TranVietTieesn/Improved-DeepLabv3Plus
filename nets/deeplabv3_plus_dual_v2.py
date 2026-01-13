"""
Dual Encoder DeepLabv3+ V2 - EfficientNet + PVTv2.

This module implements an improved Dual Encoder architecture specifically
designed for medical image segmentation (e.g., brain tumor segmentation).

Key improvements over V1:
- Uses EfficientNet (proven effective for medical imaging) instead of ConvNeXt
- EfficientNet's SE blocks provide built-in attention for local features
- PVTv2's Linear SRA provides efficient global context
- Optimized for smaller datasets (<10k images)

Architecture:
    EfficientNet-B7 (local features, SE attention)
           ↓
    Multi-level Gated Fusion
           ↑
    PVTv2-B2 (global context, transformer)
           ↓
    ASPP + Decoder → Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficientnet_backbone import EfficientNetBackbone
from .pvtv2_backbone import PVTv2Backbone
from .attention import se_block, cbam_block, eca_block


class ChannelAlignConv(nn.Module):
    """1x1 convolution to align channel dimensions."""
    
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
    Gated Fusion module to combine features from EfficientNet and PVTv2.
    
    Uses a learnable gate to adaptively blend features:
        gate = sigmoid(conv(concat(eff_feat, pvt_feat)))
        output = gate * eff_aligned + (1 - gate) * pvt_aligned
    """
    
    def __init__(self, in_channels_eff, in_channels_pvt, out_channels):
        super(GatedFusion, self).__init__()
        
        # Align channels from both branches to output channels
        self.align_eff = ChannelAlignConv(in_channels_eff, out_channels)
        self.align_pvt = ChannelAlignConv(in_channels_pvt, out_channels)
        
        # Gate network: learns which branch to emphasize
        self.gate_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, eff_feat, pvt_feat):
        """Fuse features from EfficientNet and PVTv2."""
        # Align spatial dimensions if needed
        if eff_feat.shape[2:] != pvt_feat.shape[2:]:
            pvt_feat = F.interpolate(pvt_feat, size=eff_feat.shape[2:], mode='bilinear', align_corners=True)
        
        # Align channel dimensions
        eff_aligned = self.align_eff(eff_feat)
        pvt_aligned = self.align_pvt(pvt_feat)
        
        # Compute gate
        concat = torch.cat([eff_aligned, pvt_aligned], dim=1)
        gate = self.gate_conv(concat)
        
        # Blend features
        fused = gate * eff_aligned + (1 - gate) * pvt_aligned
        
        return fused


class DualEncoderV2(nn.Module):
    """
    Dual Encoder with EfficientNet + PVTv2 and Multi-level Gated Fusion.
    """
    
    def __init__(self, efficientnet, pvtv2, low_level_out=128, high_level_out=512):
        super(DualEncoderV2, self).__init__()
        
        self.efficientnet = efficientnet
        self.pvtv2 = pvtv2
        
        # Get channel dimensions
        eff_dims = efficientnet.dims
        pvt_dims = pvtv2.dims
        
        # Low-level fusion (Stage 1: 1/4 resolution)
        self.low_level_fusion = GatedFusion(
            in_channels_eff=eff_dims[0],
            in_channels_pvt=pvt_dims[0],
            out_channels=low_level_out
        )
        
        # High-level fusion (Stage 4: 1/32 resolution)
        self.high_level_fusion = GatedFusion(
            in_channels_eff=eff_dims[3],
            in_channels_pvt=pvt_dims[3],
            out_channels=high_level_out
        )
        
        self.low_level_channels = low_level_out
        self.out_channels = high_level_out
    
    def forward(self, x):
        """Forward pass through both backbones with fusion."""
        # Get features from both backbones
        eff_feats = self.efficientnet.get_all_features(x)
        pvt_feats = self.pvtv2.get_all_features(x)
        
        # Low-level fusion
        low_level_fused = self.low_level_fusion(eff_feats[0], pvt_feats[0])
        
        # High-level fusion
        high_level_fused = self.high_level_fusion(eff_feats[3], pvt_feats[3])
        
        return low_level_fused, high_level_fused
    
    def get_output_channels(self):
        return self.low_level_channels, self.out_channels


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module with attention."""
    
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, attention_block=None):
        super(ASPP, self).__init__()
        
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
        
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        if attention_block is None:
            self.attn_block = eca_block(dim_out * 5)
        else:
            self.attn_block = attention_block(dim_out * 5)
    
    def forward(self, x):
        b, c, row, col = x.size()
        
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        
        attn = self.attn_block(feature_cat)
        result = self.conv_cat(attn)
        
        return result


class DeepLabDualV2(nn.Module):
    """
    Dual Encoder DeepLabv3+ V2 - EfficientNet + PVTv2.
    
    Optimized for medical image segmentation with smaller datasets.
    
    Args:
        num_classes: Number of output classes
        efficientnet_variant: EfficientNet variant - 'b0' to 'b7' (default 'b7')
        pvtv2_variant: PVTv2 variant - 'b0' to 'b5' (default 'b2')
        pretrained: Whether to load pretrained weights
        in_chans: Number of input channels (1 for grayscale, 3 for RGB)
        downsample_factor: Output stride (8 or 16)
        attention_block: Attention module for ASPP
        low_level_channels: Output channels for low-level fusion
        high_level_channels: Output channels for high-level fusion
    """
    
    def __init__(
        self,
        num_classes,
        efficientnet_variant='b7',
        pvtv2_variant='b2',
        pretrained=True,
        in_chans=1,
        downsample_factor=16,
        attention_block=None,
        low_level_channels=128,
        high_level_channels=512,
        decoder_channels=256  # NEW: configurable decoder size (SMP uses 1024)
    ):
        super(DeepLabDualV2, self).__init__()
        
        self.num_classes = num_classes
        self.efficientnet_variant = efficientnet_variant
        self.pvtv2_variant = pvtv2_variant
        self.downsample_factor = downsample_factor
        self.decoder_channels = decoder_channels
        
        # Create backbones
        self.efficientnet = EfficientNetBackbone(
            variant=efficientnet_variant,
            pretrained=pretrained,
            in_chans=in_chans,
            downsample_factor=downsample_factor
        )
        
        self.pvtv2 = PVTv2Backbone(
            variant=pvtv2_variant,
            pretrained=pretrained,
            in_chans=in_chans,
            downsample_factor=downsample_factor
        )
        
        # Create dual encoder with fusion
        self.backbone = DualEncoderV2(
            efficientnet=self.efficientnet,
            pvtv2=self.pvtv2,
            low_level_out=low_level_channels,
            high_level_out=high_level_channels
        )
        
        # Get output channels
        low_level_ch, high_level_ch = self.backbone.get_output_channels()
        
        # Calculate shortcut channels (proportional to decoder_channels)
        shortcut_ch = max(48, decoder_channels // 4)
        
        # ASPP module - uses decoder_channels
        self.aspp = ASPP(
            dim_in=high_level_ch,
            dim_out=decoder_channels,  # NOW CONFIGURABLE
            rate=16 // downsample_factor,
            attention_block=attention_block
        )
        
        # Decoder - uses decoder_channels
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_ch, shortcut_ch, 1),
            nn.BatchNorm2d(shortcut_ch),
            nn.ReLU(inplace=True)
        )
        
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
        
        self.cls_conv = nn.Conv2d(decoder_channels, num_classes, 1, stride=1)
    
    def forward(self, x):
        """Forward pass through Dual Encoder V2."""
        H, W = x.size(2), x.size(3)
        
        # Dual Encoder with Fusion
        low_level_features, high_level_features = self.backbone(x)
        
        # ASPP
        x = self.aspp(high_level_features)
        
        # Decoder
        low_level_features = self.shortcut_conv(low_level_features)
        
        x = F.interpolate(
            x,
            size=(low_level_features.size(2), low_level_features.size(3)),
            mode='bilinear',
            align_corners=True
        )
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        return x
    
    def get_backbone_params(self):
        """Get backbone parameters for differential learning rates."""
        return list(self.efficientnet.parameters()) + list(self.pvtv2.parameters())
    
    def get_decoder_params(self):
        """Get decoder parameters for differential learning rates."""
        return (
            list(self.backbone.low_level_fusion.parameters()) +
            list(self.backbone.high_level_fusion.parameters()) +
            list(self.aspp.parameters()) +
            list(self.shortcut_conv.parameters()) +
            list(self.cat_conv.parameters()) +
            list(self.cls_conv.parameters())
        )
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_flops(self, input_size=(1, 1, 224, 224), verbose=False):
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
    
    def get_model_info(self, input_size=(1, 1, 448, 448)):
        """Get comprehensive model information."""
        info = {
            'model': 'DeepLabDualV2 (Gated Fusion)',
            'efficientnet_variant': self.efficientnet_variant,
            'pvtv2_variant': self.pvtv2_variant,
            'num_classes': self.num_classes,
            'decoder_channels': self.decoder_channels,
            'parameters': self.count_parameters(),
            'parameters_M': self.count_parameters() / 1e6,
        }
        
        try:
            gflops, _ = self.count_flops(input_size=input_size)
            info['gflops'] = gflops
        except ImportError:
            info['gflops'] = 'N/A (install thop)'
        
        return info


def deeplabv3_plus_dual_v2(
    num_classes,
    efficientnet_variant='b7',
    pvtv2_variant='b2',
    pretrained=True,
    in_chans=1,
    downsample_factor=16,
    attention_block=None,
    low_level_channels=128,
    high_level_channels=512,
    decoder_channels=256
):
    """
    Factory function to create Dual Encoder DeepLabv3+ V2.
    
    Args:
        decoder_channels: Decoder size (256=default, 1024=match SMP)
    
    Example:
        >>> model = deeplabv3_plus_dual_v2(num_classes=4, decoder_channels=1024)
    """
    return DeepLabDualV2(
        num_classes=num_classes,
        efficientnet_variant=efficientnet_variant,
        pvtv2_variant=pvtv2_variant,
        pretrained=pretrained,
        in_chans=in_chans,
        downsample_factor=downsample_factor,
        attention_block=attention_block,
        low_level_channels=low_level_channels,
        high_level_channels=high_level_channels,
        decoder_channels=decoder_channels
    )


# Available configurations
AVAILABLE_CONFIGS = {
    'efficientnet_variants': ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
    'pvtv2_variants': ['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
    'recommended_medical': {
        'lightweight': {'efficientnet': 'b4', 'pvtv2': 'b1'},
        'balanced': {'efficientnet': 'b5', 'pvtv2': 'b2'},
        'powerful': {'efficientnet': 'b7', 'pvtv2': 'b2'},  # Default
        'heavy': {'efficientnet': 'b7', 'pvtv2': 'b3'},
    }
}
