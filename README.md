# Improved DeepLabv3+

Enhanced DeepLabv3+ implementation with multiple backbone options, attention mechanisms, and dual encoder architectures.

## Features

- **Multiple Backbones**: MobileNetV2, Xception, Swin Transformer, EfficientNet, PVTv2, ConvNeXt
- **Dual Encoder Architectures**: Combine CNN + Transformer for richer features
- **SAM-style Prompt Attention**: Learnable prompt tokens for task-specific focus
- **Attention Modules**: SE, CBAM, ECA with configurable positions
- **Configurable Decoder**: Adjustable decoder channels (256, 512, 1024)
- **Pretrained Weights**: Support via `timm` library
- **1-Channel Input**: Optimized for grayscale/medical images

## Installation

```bash
pip install torch torchvision timm thop
```

## Model Variants

### 1. Standard DeepLabv3+ (deeplabv3_plus_v2.py)

```python
from nets.deeplabv3_plus_v2 import deeplabv3_plus

model = deeplabv3_plus(
    num_classes=4, 
    backbone='swin_base',
    decoder_channels=1024,       # NEW: configurable decoder size
    attention_block='eca',
    attention_position='aspp_post'
)
```

### 2. Dual Encoder V2 - Gated Fusion (deeplabv3_plus_dual_v2.py)

Combines EfficientNet + PVTv2 with Gated Fusion mechanism.

```python
from nets.deeplabv3_plus_dual_v2 import DeepLabDualV2

model = DeepLabDualV2(
    num_classes=4,
    efficientnet_variant='b7',   # b0-b7
    pvtv2_variant='b2',          # b0-b5
    decoder_channels=512,
    low_level_channels=128,
    high_level_channels=512,
    pretrained=True,
    in_chans=1
)
```

### 3. Dual Encoder V3 - Cross-Attention (deeplabv3_plus_dual_v3.py)

Uses Cross-Attention for deeper feature interaction between backbones.

```python
from nets.deeplabv3_plus_dual_v3 import DeepLabDualV3

model = DeepLabDualV3(
    num_classes=4,
    efficientnet_variant='b7',
    pvtv2_variant='b2',
    decoder_channels=512,
    num_heads=8,                 # Cross-attention heads
    pretrained=True,
    in_chans=1
)
```

### 4. DeepLabPrompt - SAM-style (deeplabv3_plus_prompt.py)

EfficientNet + DeepLab with learnable prompt attention.

```python
from nets.deeplabv3_plus_prompt import DeepLabPrompt

model = DeepLabPrompt(
    num_classes=4,
    efficientnet_variant='b7',
    num_prompts=8,               # Learnable prompt tokens
    decoder_channels=512,
    pretrained=True,
    in_chans=1
)
```

## Model Comparison

| Model | Params | Description |
|-------|--------|-------------|
| Standard + Swin-Base | ~88M | Transformer backbone |
| Dual V2 (B7 + B2) | ~110M | EfficientNet + PVTv2 with Gated Fusion |
| Dual V3 (B7 + B2) | ~120M | Cross-Attention fusion |
| DeepLabPrompt | ~100M | SAM-style prompt attention |

## Key Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `decoder_channels` | 256, 512, 1024 | Decoder capacity (larger = more powerful) |
| `downsample_factor` | 8, 16 | Feature resolution (8 = slower but detailed) |
| `num_prompts` | 4, 8, 16 | Prompt tokens for SAM-style model |

## Backbone Options

### Standard Backbones
- `mobilenet`, `xception`
- `swin_tiny`, `swin_small`, `swin_base`

### Dual Encoder Backbones
- EfficientNet: `b0` to `b7`
- PVTv2: `b0` to `b5`
- ConvNeXt: `tiny`, `small`, `base`, `large`

## Attention Positions

| Position | Description |
|----------|-------------|
| `none` | No attention |
| `aspp_pre` | After each ASPP branch |
| `aspp_post` | After ASPP concat |
| `decoder` | After decoder fusion |

## Usage Example

```python
import torch
from nets.deeplabv3_plus_dual_v2 import DeepLabDualV2

# Create model
model = DeepLabDualV2(num_classes=4, in_chans=1)
model = model.cuda()

# Forward pass
x = torch.randn(1, 1, 512, 512).cuda()
output = model(x)
print(output.shape)  # torch.Size([1, 4, 512, 512])

# Get model info
info = model.get_model_info(input_size=(1, 1, 512, 512))
print(f"Params: {info['parameters_M']:.2f}M, GFLOPs: {info['gflops']:.2f}")
```

## Troubleshooting

- **OOM with Dual V3**: Use V2 (Gated Fusion) or reduce batch size
- **Memory Issues**: Try `decoder_channels=256` or smaller backbone
- **Training Crashes**: Adjust `num_workers` value

## License

MIT License
