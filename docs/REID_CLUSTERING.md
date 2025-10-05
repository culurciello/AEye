# ReID-based Person and Vehicle Clustering

This guide explains how to use specialized Re-Identification (ReID) models for better person and vehicle clustering, instead of generic CLIP embeddings.

## What is ReID?

Re-Identification (ReID) models are specialized neural networks trained to:
- **Person ReID**: Identify the same person across different camera views and times
- **Vehicle ReID**: Identify the same vehicle across different views, angles, and lighting

These models produce embeddings that are much better for clustering than general-purpose CLIP embeddings because they're specifically trained to distinguish between different individuals/vehicles while recognizing the same individual across different conditions.

## Benefits of ReID vs CLIP

| Feature | CLIP Embeddings | ReID Embeddings |
|---------|----------------|-----------------|
| **Designed for** | General image understanding | Person/Vehicle identification |
| **Same person/vehicle** | May cluster separately | Clusters together reliably |
| **Different angles/lighting** | Poor handling | Robust to variations |
| **Color discrimination** | Requires manual color features | Built-in |
| **Fine-grained details** | Generic features | Person-specific (clothing, gait) / Vehicle-specific (model, color) |
| **Clustering quality** | Moderate | Excellent |

## Installation

### Recommended: TorchReID (deep-person-reid)

TorchReID is a comprehensive, well-maintained ReID library with many pretrained models.

```bash
# Install TorchReID
pip install torchreid

# Or install from source for latest version
git clone https://github.com/KaiyangZhou/deep-person-reid.git
cd deep-person-reid
pip install -r requirements.txt
python setup.py develop
```

### Fallback (No installation)

If TorchReID is not available, the system will automatically fall back to a ResNet50 feature extractor (less optimal but still better than CLIP for person/vehicle clustering).

## Pretrained Models

TorchReID provides many pretrained models:

### Person ReID Models
- **OSNet** (omni-scale network): Lightweight and accurate
  - `osnet_x1_0`: Full model (2.2M params)
  - `osnet_x0_75`: 75% width (1.3M params)
  - `osnet_x0_5`: 50% width (0.6M params)
  - `osnet_x0_25`: 25% width (0.2M params)
- **ResNet50**: Classic backbone (25M params)
- **DenseNet121**: Dense connections (8M params)
- **MobileNetV2**: Mobile-optimized (3.5M params)

All models are pretrained on:
- **Market1501**: 32,668 images of 1,501 persons
- **DukeMTMC**: 36,411 images of 1,404 persons
- **MSMT17**: 126,441 images of 4,101 persons (largest dataset)

### Vehicle ReID
TorchReID primarily focuses on person ReID. For vehicles, we use the person models which generalize well to vehicle appearance.

## Usage Workflow

### Step 1: Extract ReID Features

After processing videos with AEye, extract ReID embeddings from existing detections:

```bash
# Extract features for both persons and vehicles
python scripts/extract_reid_features.py

# Or extract for specific types
python scripts/extract_reid_features.py --persons
python scripts/extract_reid_features.py --vehicles

# Use CPU if no GPU available
python scripts/extract_reid_features.py --no-gpu
```

**What this does:**
- Adds `reid_embedding` column to database
- Extracts ReID features from all person/vehicle crops
- Stores embeddings in database for clustering

**First time:** This will download pretrained models automatically (~20-50MB depending on model).

### Step 2: Generate Clustered Report with ReID

```bash
# Use ReID embeddings for clustering
python scripts/timeline.py --use-reid

# Adjust clustering parameters for ReID
python scripts/timeline.py --use-reid --reid-person-eps 0.4 --reid-vehicle-eps 0.5

# ReID works well with stricter clustering
python scripts/timeline.py --use-reid --reid-person-eps 0.3 --reid-vehicle-eps 0.4 --vehicle-min-samples 2
```

### Step 3: Compare Results

```bash
# Generate report with CLIP embeddings (baseline)
python scripts/timeline.py --output report/clip_report.html

# Generate report with ReID embeddings (improved)
python scripts/timeline.py --use-reid --output report/reid_report.html

# Compare the two reports in your browser
```

## Clustering Parameter Tuning

### ReID Clustering Parameters

ReID embeddings are more distinctive, so you can use different parameters:

**For CLIP embeddings:**
- Person eps: `0.15` (very strict, CLIP is noisy)
- Vehicle eps: `0.08-0.10` (very strict)
- Color weight: `0.5` (need manual color features)

**For ReID embeddings:**
- Person eps: `0.5` (can be more relaxed, ReID is robust)
- Vehicle eps: `0.6` (can be more relaxed)
- Color weight: N/A (color already encoded in ReID)

### Examples

```bash
# Strict clustering (more unique persons/vehicles)
python scripts/timeline.py --use-reid \
  --reid-person-eps 0.3 \
  --reid-vehicle-eps 0.4 \
  --vehicle-min-samples 2

# Relaxed clustering (group similar-looking persons/vehicles)
python scripts/timeline.py --use-reid \
  --reid-person-eps 0.7 \
  --reid-vehicle-eps 0.8

# Balanced (default)
python scripts/timeline.py --use-reid
```

## Expected Improvements

With ReID embeddings, you should see:

### For Persons
- ✅ Same person in different poses/angles → Same cluster
- ✅ Same person with different lighting → Same cluster
- ✅ Different persons with similar clothing → Different clusters
- ✅ Fewer outliers (single-detection clusters)

### For Vehicles
- ✅ Same vehicle from different angles → Same cluster
- ✅ Same vehicle in different lighting → Same cluster
- ✅ Different vehicles of same model but different color → Different clusters
- ✅ Better separation of similar-looking vehicles

## Troubleshooting

### "TorchReID not installed" error

```bash
# Install TorchReID
pip install torchreid

# Or use fallback mode (automatic, no installation needed)
# The system will use ResNet50 features instead
```

### Models download automatically

TorchReID automatically downloads pretrained models on first use. If you have network issues:

```bash
# Pre-download models manually
python -c "import torchreid; torchreid.models.build_model('osnet_x1_0', num_classes=1000, pretrained=True)"
```

### Too many clusters / Too few clusters

```bash
# If too many tiny clusters → Increase eps
python scripts/timeline.py --use-reid --reid-vehicle-eps 0.8

# If everything in one cluster → Decrease eps
python scripts/timeline.py --use-reid --reid-vehicle-eps 0.3
```

### Running out of GPU memory

```bash
# Use CPU instead
python scripts/extract_reid_features.py --no-gpu

# Or process in smaller batches (automatic in the code)
```

## Future Improvements

Potential enhancements:
- [ ] Multi-camera track association
- [ ] Temporal smoothing for video tracks
- [ ] Custom ReID model fine-tuning on your data
- [ ] Real-time ReID feature extraction during processing
- [ ] Support for other ReID models (PCB, MGN, etc.)

## References

- **TorchReID (deep-person-reid)**: https://github.com/KaiyangZhou/deep-person-reid
  - Paper: https://arxiv.org/abs/1910.10093
  - Model Zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
- **OSNet Paper**: https://arxiv.org/abs/1905.00953
- **Market1501 Dataset**: http://zheng-lab.cecs.anu.edu.au/Project/project_reid.html
- **DukeMTMC Dataset**: https://exposing.ai/duke_mtmc/
- **MSMT17 Dataset**: https://www.pkuvmc.com/dataset.html
