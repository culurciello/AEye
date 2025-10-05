# TorchReID Models Guide

This guide explains the different ReID models available in TorchReID and when to use them.

## Available Models

### OSNet (Omni-Scale Network) - **Recommended**

OSNet is the most efficient and accurate model for person ReID.

**Variants:**
- `osnet_x1_0` - Full model (2.2M params) - **Best accuracy**
- `osnet_x0_75` - 75% width (1.3M params) - Good balance
- `osnet_x0_5` - 50% width (0.6M params) - Fast inference
- `osnet_x0_25` - 25% width (0.2M params) - Very fast, lower accuracy

**When to use:**
- Default choice for person ReID
- Best balance of speed and accuracy
- Works well for both persons and vehicles

### ResNet50

Classic CNN backbone, widely used.

**Specs:**
- 25M parameters
- Slower than OSNet
- Good accuracy

**When to use:**
- When you need proven reliability
- When OSNet models don't work well
- For transfer learning experiments

### DenseNet121

Dense connections between layers.

**Specs:**
- 8M parameters
- Good feature reuse
- Moderate speed

**When to use:**
- Alternative to ResNet50
- When you want different architecture

### MobileNetV2

Optimized for mobile/edge devices.

**Specs:**
- 3.5M parameters
- Very fast inference
- Lower accuracy

**When to use:**
- Edge deployment
- Real-time processing needed
- Resource-constrained environments

## Changing Models

To use a different model, edit `lib/reid_extractor.py`:

```python
# In init_model() method

# Change this line (around line 40):
model_name = 'osnet_x1_0'  # Current default

# To one of these:
model_name = 'osnet_x0_75'  # Faster, slightly less accurate
model_name = 'osnet_x0_5'   # Much faster, good accuracy
model_name = 'resnet50'     # Classic backbone
model_name = 'densenet121'  # Dense connections
model_name = 'mobilenetv2' # Mobile-optimized
```

## Performance Comparison

| Model | Speed | Accuracy | Memory | Best For |
|-------|-------|----------|--------|----------|
| `osnet_x1_0` | Fast | Excellent | 2.2M | **General use (default)** |
| `osnet_x0_75` | Faster | Very Good | 1.3M | Balanced |
| `osnet_x0_5` | Very Fast | Good | 0.6M | Speed priority |
| `osnet_x0_25` | Ultra Fast | Moderate | 0.2M | Edge devices |
| `resnet50` | Slow | Excellent | 25M | Accuracy priority |
| `densenet121` | Moderate | Very Good | 8M | Alternative backbone |
| `mobilenetv2` | Very Fast | Good | 3.5M | Mobile deployment |

## Recommendations

### For Most Users
```python
model_name = 'osnet_x1_0'  # Default - best balance
```

### For Speed Priority
```python
model_name = 'osnet_x0_5'  # 2x faster, still good accuracy
```

### For Accuracy Priority
```python
model_name = 'resnet50'  # Proven accuracy, but slower
```

### For Edge Devices
```python
model_name = 'osnet_x0_25'  # Smallest model
# or
model_name = 'mobilenetv2'  # Mobile-optimized
```

## Testing Different Models

To compare different models on your data:

```bash
# 1. Extract features with default model (osnet_x1_0)
python scripts/extract_reid_features.py

# 2. Generate report
python scripts/timeline.py --use-reid --output report/osnet_x1_0.html

# 3. Change model in lib/reid_extractor.py to osnet_x0_5

# 4. Clear existing ReID embeddings
sqlite3 data/db/detections.db "UPDATE object_detections SET reid_embedding = NULL"

# 5. Extract with new model
python scripts/extract_reid_features.py

# 6. Generate new report
python scripts/timeline.py --use-reid --output report/osnet_x0_5.html

# 7. Compare the two reports
```

## Model Download

Models are downloaded automatically on first use from:
- GitHub: https://github.com/KaiyangZhou/deep-person-reid
- Cache location: `~/.cache/torch/hub/checkpoints/`

To pre-download models:

```python
import torchreid

# Download specific model
torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)
torchreid.models.build_model('osnet_x0_75', num_classes=1000, pretrained=True)
```

## Advanced: Custom Models

You can also use custom trained models:

```python
# In init_model()
self.model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1000,
    pretrained=False  # Don't load pretrained weights
)

# Load your custom weights
checkpoint = torch.load('path/to/your/model.pth')
self.model.load_state_dict(checkpoint)
```

## References

- **TorchReID Documentation**: https://kaiyangzhou.github.io/deep-person-reid/
- **Model Zoo**: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
- **OSNet Paper**: https://arxiv.org/abs/1905.00953
