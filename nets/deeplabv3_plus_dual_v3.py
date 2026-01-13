"""
Dual Encoder DeepLabv3+ V3 - EfficientNet + PVTv2 with Cross-Attention Fusion.

This version uses Cross-Attention Fusion instead of Gated Fusion,
allowing the two backbones to "communicate" and learn correlations between features.

Key differences from V2:
- Cross-Attention: EfficientNet queries PVTv2's features and vice versa
- Bidirectional information flow between CNN and Transformer
- Deeper feature interaction at multiple levels
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


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion module.
    
    Enables bidirectional attention between EfficientNet and PVTv2 features:
    - EfficientNet features query PVTv2 features (CNN asks Transformer)
    - PVTv2 features query EfficientNet features (Transformer asks CNN)
    
    This allows the model to learn which features from each backbone
    are most relevant for each spatial location.
    
    Args:
        in_channels_eff: Channels from EfficientNet
        in_channels_pvt: Channels from PVTv2
        out_channels: Output channels after fusion
        num_heads: Number of attention heads
        dropout: Dropout rate for attention
    """
    
    def __init__(self, in_channels_eff, in_channels_pvt, out_channels, num_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.head_dim = out_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Align channels to out_channels
        self.align_eff = ChannelAlignConv(in_channels_eff, out_channels)
        self.align_pvt = ChannelAlignConv(in_channels_pvt, out_channels)
        
        # Cross-attention: EfficientNet → PVTv2 (CNN queries Transformer)
        self.q_eff = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.k_pvt = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.v_pvt = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        
        # Cross-attention: PVTv2 → EfficientNet (Transformer queries CNN)
        self.q_pvt = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.k_eff = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.v_eff = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        
        # Output projection
        self.proj = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (applied after reshape)
        self.norm_eff = nn.LayerNorm(out_channels)
        self.norm_pvt = nn.LayerNorm(out_channels)
        
        # Feed-forward network for refinement
        self.ffn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.Dropout(dropout)
        )
        self.norm_out = nn.BatchNorm2d(out_channels)
    
    def _attention(self, q, k, v, B, H, W):
        """Compute multi-head attention."""
        # Reshape for multi-head attention: (B, C, H, W) -> (B, heads, H*W, head_dim)
        N = H * W
        q = q.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)  # (B, heads, N, head_dim)
        k = k.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        
        # Attention: (B, heads, N, head_dim) @ (B, heads, head_dim, N) -> (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values: (B, heads, N, N) @ (B, heads, N, head_dim) -> (B, heads, N, head_dim)
        out = attn @ v
        
        # Reshape back: (B, heads, N, head_dim) -> (B, C, H, W)
        out = out.permute(0, 1, 3, 2).reshape(B, self.out_channels, H, W)
        
        return out
    
    def forward(self, eff_feat, pvt_feat):
        """
        Cross-attention fusion between EfficientNet and PVTv2 features.
        
        Args:
            eff_feat: Features from EfficientNet
            pvt_feat: Features from PVTv2
            
        Returns:
            Fused features with cross-attention interaction
        """
        # Align spatial dimensions
        if eff_feat.shape[2:] != pvt_feat.shape[2:]:
            pvt_feat = F.interpolate(pvt_feat, size=eff_feat.shape[2:], mode='bilinear', align_corners=True)
        
        B, _, H, W = eff_feat.shape
        
        # Align channels
        eff_aligned = self.align_eff(eff_feat)
        pvt_aligned = self.align_pvt(pvt_feat)
        
        # Cross-attention 1: EfficientNet queries PVTv2
        # "What information from the Transformer is relevant for CNN features?"
        q1 = self.q_eff(eff_aligned)
        k1 = self.k_pvt(pvt_aligned)
        v1 = self.v_pvt(pvt_aligned)
        eff_attended = self._attention(q1, k1, v1, B, H, W)
        eff_attended = eff_aligned + eff_attended  # Residual connection
        
        # Cross-attention 2: PVTv2 queries EfficientNet
        # "What information from the CNN is relevant for Transformer features?"
        q2 = self.q_pvt(pvt_aligned)
        k2 = self.k_eff(eff_aligned)
        v2 = self.v_eff(eff_aligned)
        pvt_attended = self._attention(q2, k2, v2, B, H, W)
        pvt_attended = pvt_aligned + pvt_attended  # Residual connection
        
        # Combine both attended features
        combined = torch.cat([eff_attended, pvt_attended], dim=1)
        fused = self.proj(combined)
        
        # FFN refinement with residual
        fused = fused + self.ffn(fused)
        fused = self.norm_out(fused)
        
        return fused


class DualEncoderV3(nn.Module):
    """Dual Encoder with Cross-Attention Fusion."""
    
    def __init__(self, efficientnet, pvtv2, low_level_out=128, high_level_out=512, num_heads=8):
        super(DualEncoderV3, self).__init__()
        
        self.efficientnet = efficientnet
        self.pvtv2 = pvtv2
        
        # Get channel dimensions
        eff_dims = efficientnet.dims
        pvt_dims = pvtv2.dims
        
        # Cross-attention fusion at low-level (Stage 1)
        self.low_level_fusion = CrossAttentionFusion(
            in_channels_eff=eff_dims[0],
            in_channels_pvt=pvt_dims[0],
            out_channels=low_level_out,
            num_heads=min(num_heads, low_level_out // 8)
        )
        
        # Cross-attention fusion at high-level (Stage 4)
        self.high_level_fusion = CrossAttentionFusion(
            in_channels_eff=eff_dims[3],
            in_channels_pvt=pvt_dims[3],
            out_channels=high_level_out,
            num_heads=num_heads
        )
        
        self.low_level_channels = low_level_out
        self.out_channels = high_level_out
    
    def forward(self, x):
        """Forward pass with cross-attention fusion."""
        # Get features from both backbones
        eff_feats = self.efficientnet.get_all_features(x)
        pvt_feats = self.pvtv2.get_all_features(x)
        
        # Cross-attention fusion at low-level
        low_level_fused = self.low_level_fusion(eff_feats[0], pvt_feats[0])
        
        # Cross-attention fusion at high-level
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


class DeepLabDualV3(nn.Module):
    """
    Dual Encoder DeepLabv3+ V3 - EfficientNet + PVTv2 with Cross-Attention.
    
    Uses Cross-Attention Fusion for deeper feature interaction between
    CNN and Transformer branches.
    
    Args:
        num_classes: Number of output classes
        efficientnet_variant: EfficientNet variant - 'b0' to 'b7'
        pvtv2_variant: PVTv2 variant - 'b0' to 'b5'
        pretrained: Whether to load pretrained weights
        in_chans: Number of input channels
        downsample_factor: Output stride (8 or 16)
        attention_block: Attention module for ASPP
        low_level_channels: Output channels for low-level fusion
        high_level_channels: Output channels for high-level fusion
        num_heads: Number of attention heads for cross-attention
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
        decoder_channels=256,  # NEW: configurable decoder size (SMP uses 1024)
        num_heads=8
    ):
        super(DeepLabDualV3, self).__init__()
        
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
        
        # Create dual encoder with cross-attention fusion
        self.backbone = DualEncoderV3(
            efficientnet=self.efficientnet,
            pvtv2=self.pvtv2,
            low_level_out=low_level_channels,
            high_level_out=high_level_channels,
            num_heads=num_heads
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
        """Forward pass through Dual Encoder V3 with Cross-Attention."""
        H, W = x.size(2), x.size(3)
        
        # Dual Encoder with Cross-Attention Fusion
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
            'model': 'DeepLabDualV3 (Cross-Attention)',
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


def deeplabv3_plus_dual_v3(
    num_classes,
    efficientnet_variant='b7',
    pvtv2_variant='b2',
    pretrained=True,
    in_chans=1,
    downsample_factor=16,
    attention_block=None,
    low_level_channels=128,
    high_level_channels=512,
    decoder_channels=256,
    num_heads=8
):
    """
    Factory function to create Dual Encoder DeepLabv3+ V3 with Cross-Attention.
    
    Args:
        decoder_channels: Decoder size (256=default, 1024=match SMP)
    
    Example:
        >>> model = deeplabv3_plus_dual_v3(num_classes=4, decoder_channels=1024)
        >>> x = torch.randn(1, 1, 512, 512)
        >>> output = model(x)
    """
    return DeepLabDualV3(
        num_classes=num_classes,
        efficientnet_variant=efficientnet_variant,
        pvtv2_variant=pvtv2_variant,
        pretrained=pretrained,
        in_chans=in_chans,
        downsample_factor=downsample_factor,
        attention_block=attention_block,
        low_level_channels=low_level_channels,
        high_level_channels=high_level_channels,
        decoder_channels=decoder_channels,
        num_heads=num_heads
    )


# Configurations
AVAILABLE_CONFIGS = {
    'efficientnet_variants': ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'],
    'pvtv2_variants': ['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
    'recommended': {
        'balanced': {'efficientnet': 'b7', 'pvtv2': 'b2', 'num_heads': 8},
        'powerful': {'efficientnet': 'b7', 'pvtv2': 'b3', 'num_heads': 8},
    }
}
