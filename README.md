# Improved DeepLabv3+

Enhanced DeepLabv3+ implementation with multiple backbone options and attention mechanisms.

## Features

- **Multiple Backbones**: MobileNetV2, Xception, Swin Transformer (Tiny/Small/Base)
- **Attention Modules**: SE, CBAM, ECA with configurable positions
- **Pretrained Weights**: Support for ImageNet pretrained weights via `timm`
- **1-Channel Input**: Optimized for grayscale/single-channel images

## Installation

```bash
pip install torch torchvision timm
```

## Quick Start

```python
from nets.deeplabv3_plus_v2 import DeepLab, deeplabv3_plus

# Basic usage with Swin-Tiny backbone
model = DeepLab(
    num_classes=2, 
    backbone='swin_tiny',  # 'mobilenet', 'xception', 'swin_tiny', 'swin_small', 'swin_base'
    pretrained=True
)

# With attention
model = deeplabv3_plus(
    num_classes=2, 
    backbone='swin_base',
    attention_block='eca',          # 'se', 'cbam', 'eca'
    attention_position='aspp_post'  # 'none', 'aspp_pre', 'aspp_post', 'decoder'
)
```

## Backbone Comparison

| Backbone | Params | GFLOPs | Equivalent |
|----------|--------|--------|------------|
| MobileNetV2 | 3.5M | 0.3 | - |
| Xception | 23M | 8.1 | - |
| Swin-Tiny | 29M | 4.5 | ResNet-50 |
| Swin-Small | 50M | 8.7 | ResNet-101 |
| Swin-Base | 88M | 15.4 | ViT-B |

## Attention Positions

| Position | Description |
|----------|-------------|
| `none` | No attention |
| `aspp_pre` | After each ASPP branch, before concat |
| `aspp_post` | After ASPP concat |
| `decoder` | After decoder feature fusion |

## Architecture

![Architecture Diagram](https://github.com/vitant-lang/improved-Deeplabv3-/assets/75409802/16c1296e-d669-4c75-91f7-601bc02965d3)

## Troubleshooting

- **Memory Issues**: Reduce batch size or try Swin-Tiny instead of Base
- **Training Crashes**: Adjust `num_workers` value (CTRL+F in files)

## License

MIT License
