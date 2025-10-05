# ReID Clustering - Quick Start Guide

Get better person and vehicle clustering in 3 steps!

## Quick Setup

### 1. Install TorchReID (Optional but Recommended)

```bash
pip install torchreid
```

**Note:** If you skip this step, the system will use a fallback ResNet50 model (less optimal).

### 2. Extract ReID Features from Your Existing Data

```bash
# Extract ReID features for all person and vehicle detections
python scripts/extract_reid_features.py

# This will:
# - Add reid_embedding column to your database
# - Extract specialized ReID features from all person/vehicle crops
# - Store them in the database
```

**First run:** Downloads pretrained models automatically (~20-50MB)

### 3. Generate Clustered Report with ReID

```bash
# Use ReID embeddings for clustering
python scripts/timeline.py --use-reid

# Your report will be at: report/detection_report.html
```

## Before vs After Comparison

### Before (CLIP embeddings + color features)
```bash
python scripts/timeline.py --output report/before_clip.html
```

**Issues:**
- Same vehicle in different lighting → different clusters
- Similar-looking vehicles → grouped together
- Many outliers (single detection clusters)
- Color features needed manual tuning

### After (ReID embeddings)
```bash
python scripts/timeline.py --use-reid --output report/after_reid.html
```

**Improvements:**
- Same vehicle across different views → same cluster
- Different vehicles of same model → separated by color
- Fewer outliers
- No manual color tuning needed

## Common Commands

```bash
# Extract ReID features (do this once after processing videos)
python scripts/extract_reid_features.py

# Generate report with ReID (default settings)
python scripts/timeline.py --use-reid

# Stricter clustering (more unique persons/vehicles)
python scripts/timeline.py --use-reid \
  --reid-person-eps 0.3 \
  --reid-vehicle-eps 0.4

# Relaxed clustering (group similar-looking ones)
python scripts/timeline.py --use-reid \
  --reid-person-eps 0.7 \
  --reid-vehicle-eps 0.8
```

## What Changed?

| Aspect | Before (CLIP) | After (ReID) |
|--------|---------------|--------------|
| **Person clustering** | Poor - same person split across clusters | Excellent - same person stays together |
| **Vehicle clustering** | Moderate - needs manual color tuning | Excellent - automatic color + appearance |
| **Parameter tuning** | Complex - many parameters | Simple - just eps value |
| **Robustness** | Sensitive to lighting/angle | Robust to variations |

## Workflow Integration

### Current Workflow
```bash
# 1. Process videos
python process.py

# 2. Generate report
python scripts/timeline.py
```

### New ReID Workflow
```bash
# 1. Process videos
python process.py

# 2. Extract ReID features (once)
python scripts/extract_reid_features.py

# 3. Generate report with ReID
python scripts/timeline.py --use-reid
```

**Note:** You only need to run `extract_reid_features.py` once after processing. Future reports can use `--use-reid` directly.

## Troubleshooting

### "No detections found with reid_embedding"
→ Run `python scripts/extract_reid_features.py` first

### "TorchReID not installed"
→ Either install it (`pip install torchreid`) or use fallback (automatic)

### Clusters look wrong
→ Adjust eps parameters:
- Too many small clusters → Increase eps (try 0.7-0.8)
- Everything in one cluster → Decrease eps (try 0.3-0.4)

## Need Help?

See full documentation: [REID_CLUSTERING.md](REID_CLUSTERING.md)
