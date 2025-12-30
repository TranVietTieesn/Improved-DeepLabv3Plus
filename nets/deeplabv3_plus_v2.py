"""
DeepLabv3+ with Swin Transformer Backbone Support
==================================================
Extended DeepLabv3+ implementation supporting:
- Original backbones: mobilenet, xception
- New backbones: swin_tiny, swin_small, swin_base

Features:
- Configurable attention positions: none, aspp_pre, aspp_post, decoder
- Pretrained weight loading via timm
- 1-channel input support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .xception import xception
from .mobilenetv2 import mobilenetv2
from .attention import se_block, cbam_block, eca_block

# Attention blocks registry
ATTENTION_BLOCKS = {
    'se': se_block,
    'cbam': cbam_block,
    'eca': eca_block,
}


class MobileNetV2Backbone(nn.Module):
    """MobileNetV2 backbone wrapper."""
    
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2Backbone, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
        # Output channels
        self.out_channels = 320
        self.low_level_channels = 24

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module with optional attention.
    
    Args:
        dim_in: Input channels
        dim_out: Output channels
        rate: Dilation rate multiplier
        bn_mom: BatchNorm momentum
        attention_block: Attention block class (se_block, cbam_block, eca_block)
        attention_position: Where to apply attention ('none', 'aspp_pre', 'aspp_post')
    """
    
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1, 
                 attention_block=None, attention_position='none'):
        super(ASPP, self).__init__()
        
        self.attention_position = attention_position
        
        # ASPP branches
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
        
        # Global average pooling branch
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # Concat conv
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # Attention blocks based on position
        self.attn_pre = None
        self.attn_post = None
        
        if attention_block is not None:
            if attention_position == 'aspp_pre':
                # Attention on each branch output
                self.attn_pre = attention_block(dim_out)
            elif attention_position == 'aspp_post':
                # Attention after concatenation
                self.attn_post = attention_block(dim_out * 5)

    def forward(self, x):
        b, c, row, col = x.size()
        
        # ASPP branches
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        
        # Global pooling branch
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        
        # Apply pre-concat attention if enabled
        if self.attn_pre is not None:
            conv1x1 = self.attn_pre(conv1x1)
            conv3x3_1 = self.attn_pre(conv3x3_1)
            conv3x3_2 = self.attn_pre(conv3x3_2)
            conv3x3_3 = self.attn_pre(conv3x3_3)
            global_feature = self.attn_pre(global_feature)
        
        # Concatenate
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        
        # Apply post-concat attention if enabled
        if self.attn_post is not None:
            feature_cat = self.attn_post(feature_cat)
        
        result = self.conv_cat(feature_cat)
        return result


class DeepLab(nn.Module):
    """
    DeepLabv3+ with support for multiple backbones and configurable attention.
    
    Args:
        num_classes: Number of output classes
        backbone: Backbone network ('mobilenet', 'xception', 'swin_tiny', 'swin_small', 'swin_base')
        pretrained: Whether to load pretrained backbone weights
        downsample_factor: Output stride (8 or 16)
        attention_block: Attention type ('se', 'cbam', 'eca') or None
        attention_position: Where to apply attention:
            - 'none': No attention
            - 'aspp_pre': After each ASPP branch, before concat
            - 'aspp_post': After ASPP concat
            - 'decoder': After decoder concat (low-level + upsampled features)
    """
    
    def __init__(self, num_classes, backbone="mobilenet", pretrained=True, 
                 downsample_factor=16, attention_block=None, attention_position='none'):
        super(DeepLab, self).__init__()
        
        self.backbone_name = backbone
        self.attention_position = attention_position
        
        # Get attention block class
        attn_block_class = None
        if attention_block is not None and attention_position != 'none':
            if isinstance(attention_block, str):
                attn_block_class = ATTENTION_BLOCKS.get(attention_block, eca_block)
            else:
                attn_block_class = attention_block
        
        # Initialize backbone
        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
            
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2Backbone(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
            
        elif backbone.startswith("swin_"):
            from .swin_transformer import swin_transformer
            variant = backbone.replace("swin_", "")  # 'tiny', 'small', 'base'
            self.backbone = swin_transformer(
                variant=variant, 
                pretrained=pretrained, 
                in_chans=1,
                downsample_factor=downsample_factor
            )
            low_level_channels, in_channels = self.backbone.get_output_channels()
            
        else:
            raise ValueError(f'Unsupported backbone - `{backbone}`. '
                           f'Use mobilenet, xception, swin_tiny, swin_small, swin_base.')

        # ASPP module
        aspp_attention_pos = attention_position if attention_position in ['aspp_pre', 'aspp_post'] else 'none'
        self.aspp = ASPP(
            dim_in=in_channels, 
            dim_out=256, 
            rate=16 // downsample_factor,
            attention_block=attn_block_class,
            attention_position=aspp_attention_pos
        )

        # Low-level feature projection
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder
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
        
        # Decoder attention (if position is 'decoder')
        self.decoder_attention = None
        if attention_position == 'decoder' and attn_block_class is not None:
            self.decoder_attention = attn_block_class(48 + 256)
        
        # Final classification
        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        
        # ==========================================
        # ENCODER
        # ==========================================
        low_level_features, high_level_features = self.backbone(x)
        
        # ASPP (with optional attention inside)
        x = self.aspp(high_level_features)

        # ==========================================
        # DECODER
        # ==========================================
        # Process low-level features
        low_level_features = self.shortcut_conv(low_level_features)

        # Upsample and concatenate
        x = F.interpolate(x, size=(low_level_features.size(2), low_level_features.size(3)), 
                         mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_features), dim=1)
        
        # Apply decoder attention if enabled
        if self.decoder_attention is not None:
            x = self.decoder_attention(x)
        
        x = self.cat_conv(x)
        x = self.cls_conv(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        
        return x


def deeplabv3_plus(num_classes, backbone='mobilenet', pretrained=True, 
                   downsample_factor=16, attention_block=None, attention_position='none'):
    """
    Factory function to create DeepLabv3+ model.
    
    Args:
        num_classes: Number of output segmentation classes
        backbone: One of 'mobilenet', 'xception', 'swin_tiny', 'swin_small', 'swin_base'
        pretrained: Load pretrained backbone weights
        downsample_factor: Output stride (8 or 16)
        attention_block: Attention type ('se', 'cbam', 'eca') or None
        attention_position: 'none', 'aspp_pre', 'aspp_post', or 'decoder'
        
    Returns:
        DeepLab model instance
        
    Examples:
        >>> # Basic usage with Swin-Tiny
        >>> model = deeplabv3_plus(num_classes=2, backbone='swin_tiny')
        
        >>> # With attention after ASPP
        >>> model = deeplabv3_plus(num_classes=2, backbone='swin_base', 
        ...                        attention_block='eca', attention_position='aspp_post')
    """
    return DeepLab(
        num_classes=num_classes,
        backbone=backbone,
        pretrained=pretrained,
        downsample_factor=downsample_factor,
        attention_block=attention_block,
        attention_position=attention_position
    )
