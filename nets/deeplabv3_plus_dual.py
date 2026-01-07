"""
Dual Encoder DeepLabv3+ Model.

This module implements DeepLabv3+ with a Dual Encoder architecture,
combining ConvNeXt (modern CNN) and PVTv2 (efficient Vision Transformer)
with Multi-level Gated Fusion.

Features:
- Complementary feature extraction from CNN + Transformer
- Multi-level Gated Fusion at low and high levels
- Flexible input sizes (224, 256, 384, 448, 512, etc.)
- Configurable backbone variants
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convnext_backbone import ConvNeXtBackbone
from .pvtv2_backbone import PVTv2Backbone
from .dual_encoder_fusion import DualEncoderWithFusion
from .attention import se_block, cbam_block, eca_block


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module with attention.
    
    Uses dilated convolutions with different rates to capture multi-scale context.
    
    Args:
        dim_in: Input channels from backbone
        dim_out: Output channels for each branch
        rate: Base dilation rate
        bn_mom: BatchNorm momentum
        attention_block: Attention module to apply (se_block, cbam_block, eca_block)
    """
    
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, attention_block=None):
        super(ASPP, self).__init__()
        
        # 1x1 convolution branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        # 3x3 convolution with rate 6
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        # 3x3 convolution with rate 12
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        # 3x3 convolution with rate 18
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        # Global average pooling branch
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)
        
        # Concatenation and 1x1 conv
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        
        # Attention block
        if attention_block is None:
            self.attn_block = eca_block(dim_out * 5)
        else:
            self.attn_block = attention_block(dim_out * 5)
    
    def forward(self, x):
        b, c, row, col = x.size()
        
        # Apply all branches
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        
        # Global average pooling
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        
        # Concatenate all branches
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        
        # Apply attention and 1x1 conv
        attn = self.attn_block(feature_cat)
        result = self.conv_cat(attn)
        
        return result


class DeepLabDual(nn.Module):
    """
    Dual Encoder DeepLabv3+ Model.
    
    Combines ConvNeXt and PVTv2 backbones with Multi-level Gated Fusion
    for enhanced semantic segmentation.
    
    Args:
        num_classes: Number of output classes
        convnext_variant: ConvNeXt variant - 'tiny', 'small', 'base', 'large'
        pvtv2_variant: PVTv2 variant - 'b0', 'b1', 'b2', 'b3', 'b4', 'b5'
        pretrained: Whether to load pretrained weights
        in_chans: Number of input channels (1 for grayscale, 3 for RGB)
        downsample_factor: Output stride for backbones (8 or 16)
        attention_block: Attention module for ASPP (se_block, cbam_block, eca_block)
        low_level_channels: Output channels for low-level fusion
        high_level_channels: Output channels for high-level fusion
    """
    
    def __init__(
        self,
        num_classes,
        convnext_variant='tiny',
        pvtv2_variant='b2',
        pretrained=True,
        in_chans=1,
        downsample_factor=16,
        attention_block=None,
        low_level_channels=128,
        high_level_channels=512
    ):
        super(DeepLabDual, self).__init__()
        
        self.num_classes = num_classes
        self.convnext_variant = convnext_variant
        self.pvtv2_variant = pvtv2_variant
        self.downsample_factor = downsample_factor
        
        # Create backbones
        self.convnext = ConvNeXtBackbone(
            variant=convnext_variant,
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
        self.backbone = DualEncoderWithFusion(
            convnext_backbone=self.convnext,
            pvtv2_backbone=self.pvtv2,
            low_level_out=low_level_channels,
            high_level_out=high_level_channels
        )
        
        # Get output channels
        low_level_ch, high_level_ch = self.backbone.get_output_channels()
        
        # ASPP module
        self.aspp = ASPP(
            dim_in=high_level_ch,
            dim_out=256,
            rate=16 // downsample_factor,
            attention_block=attention_block
        )
        
        # Decoder: process low-level features
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_ch, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: combine ASPP output with low-level features
        self.cat_conv = nn.Sequential(
            nn.Conv2d(48 + 256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.1),
        )
        
        # Final classification layer
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)
    
    def forward(self, x):
        """
        Forward pass through Dual Encoder DeepLabv3+.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output segmentation map (B, num_classes, H, W)
        """
        H, W = x.size(2), x.size(3)
        
        # ==========================================
        # DUAL ENCODER WITH FUSION
        # ==========================================
        low_level_features, high_level_features = self.backbone(x)
        
        # ==========================================
        # ASPP MODULE
        # ==========================================
        x = self.aspp(high_level_features)
        
        # ==========================================
        # DECODER
        # ==========================================
        # Process low-level features
        low_level_features = self.shortcut_conv(low_level_features)
        
        # Upsample ASPP output and concatenate with low-level features
        x = F.interpolate(
            x,
            size=(low_level_features.size(2), low_level_features.size(3)),
            mode='bilinear',
            align_corners=True
        )
        x = self.cat_conv(torch.cat((x, low_level_features), dim=1))
        
        # Final classification
        x = self.cls_conv(x)
        
        # Upsample to original resolution
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        return x
    
    def get_backbone_params(self):
        """Get parameters from backbone (for differential learning rates)."""
        return list(self.convnext.parameters()) + list(self.pvtv2.parameters())
    
    def get_decoder_params(self):
        """Get parameters from decoder (for differential learning rates)."""
        return (
            list(self.backbone.fusion.parameters()) +
            list(self.aspp.parameters()) +
            list(self.shortcut_conv.parameters()) +
            list(self.cat_conv.parameters()) +
            list(self.cls_conv.parameters())
        )
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_flops(self, input_size=(1, 1, 224, 224), verbose=False):
        """
        Count FLOPs (Floating Point Operations) for the model.
        
        Requires 'thop' library: pip install thop
        
        Note: This method creates a copy of the model for profiling to avoid
        interfering with training. Call this BEFORE training, not during.
        
        Args:
            input_size: Tuple of (batch, channels, height, width)
            verbose: Whether to print per-layer details
            
        Returns:
            Tuple of (gflops, params) where gflops is in GFLOPs
            
        Example:
            >>> model = DeepLabDual(num_classes=2, pretrained=False)
            >>> gflops, params = model.count_flops(input_size=(1, 1, 448, 448))
            >>> print(f"GFLOPs: {gflops:.2f}, Params: {params/1e6:.2f}M")
        """
        try:
            from thop import profile
        except ImportError:
            raise ImportError(
                "thop library is required for FLOPs counting. "
                "Install it with: pip install thop"
            )
        
        import copy
        
        # Create a copy of the model for profiling to avoid modifying the original
        device = next(self.parameters()).device
        model_copy = copy.deepcopy(self)
        model_copy.eval()
        
        dummy_input = torch.randn(input_size).to(device)
        
        with torch.no_grad():
            flops, params = profile(model_copy, inputs=(dummy_input,), verbose=verbose)
        
        # Clean up the copy
        del model_copy
        
        # Convert to GFLOPs
        gflops = flops / 1e9
        
        return gflops, params
    
    def get_model_info(self, input_size=(1, 1, 448, 448)):
        """
        Get comprehensive model information including params and FLOPs.
        
        Args:
            input_size: Tuple of (batch, channels, height, width)
            
        Returns:
            Dictionary with model information
        """
        info = {
            'convnext_variant': self.convnext_variant,
            'pvtv2_variant': self.pvtv2_variant,
            'num_classes': self.num_classes,
            'parameters': self.count_parameters(),
            'parameters_M': self.count_parameters() / 1e6,
        }
        
        try:
            gflops, _ = self.count_flops(input_size=input_size)
            info['gflops'] = gflops
            info['input_size'] = input_size
        except ImportError:
            info['gflops'] = 'N/A (install thop)'
        
        return info


def deeplabv3_plus_dual(
    num_classes,
    convnext_variant='tiny',
    pvtv2_variant='b2',
    pretrained=True,
    in_chans=1,
    downsample_factor=16,
    attention_block=None
):
    """
    Factory function to create Dual Encoder DeepLabv3+.
    
    Args:
        num_classes: Number of output classes
        convnext_variant: ConvNeXt variant - 'tiny', 'small', 'base', 'large'
        pvtv2_variant: PVTv2 variant - 'b0', 'b1', 'b2', 'b3', 'b4', 'b5'
        pretrained: Whether to load pretrained weights
        in_chans: Number of input channels (1 for grayscale, 3 for RGB)
        downsample_factor: Output stride (8 or 16)
        attention_block: Attention module for ASPP
        
    Returns:
        DeepLabDual model
        
    Example:
        >>> model = deeplabv3_plus_dual(num_classes=2, convnext_variant='tiny', pvtv2_variant='b2')
        >>> x = torch.randn(1, 1, 448, 448)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 2, 448, 448])
    """
    return DeepLabDual(
        num_classes=num_classes,
        convnext_variant=convnext_variant,
        pvtv2_variant=pvtv2_variant,
        pretrained=pretrained,
        in_chans=in_chans,
        downsample_factor=downsample_factor,
        attention_block=attention_block
    )


# Available configurations for easy reference
AVAILABLE_CONFIGS = {
    'convnext_variants': ['tiny', 'small', 'base', 'large'],
    'pvtv2_variants': ['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
    'attention_blocks': ['se_block', 'cbam_block', 'eca_block', None],
    'recommended': {
        'lightweight': {'convnext': 'tiny', 'pvtv2': 'b1'},
        'balanced': {'convnext': 'tiny', 'pvtv2': 'b2'},
        'powerful': {'convnext': 'small', 'pvtv2': 'b3'},
        'heavy': {'convnext': 'base', 'pvtv2': 'b4'},
    }
}
