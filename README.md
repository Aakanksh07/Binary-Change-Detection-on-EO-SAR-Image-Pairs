# Binary Change Detection on EO-SAR Image Pairs

Binary pixel-level change detection on paired Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite imagery using a UNet architecture with a pretrained ResNet34 encoder. Each pixel is classified as **Changed** (1) or **Unchanged** (0) between a pre-event and post-event image pair.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (tested on NVIDIA T4, 16 GB VRAM)

All dependencies with pinned versions:

```
torch==2.6.0
torchvision==0.21.0
segmentation-models-pytorch==0.3.3
albumentations==1.3.1
rasterio==1.3.9
huggingface_hub==0.23.0
numpy==1.26.4
scikit-learn==1.4.2
matplotlib==3.8.4
pyyaml==6.0.1
```

Install via:
```bash
pip install -r requirements.txt
```

---

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

Or with conda:
```bash
conda create -n galaxeye python=3.10
conda activate galaxeye
pip install -r requirements.txt
```

---

## Dataset Structure

Download the dataset from [HuggingFace](https://huggingface.co/datasets/doron333/change-detection-dataset) and place it as follows:

```
data/
├── train/
│   ├── pre-event/
│   │   ├── image_001.tif
│   │   └── ...
│   ├── post-event/
│   │   ├── image_001.tif
│   │   └── ...
│   └── target/
│       ├── image_001.tif
│       └── ...
├── val/
│   ├── pre-event/
│   ├── post-event/
│   └── target/
└── test/
    ├── pre-event/
    ├── post-event/
    └── target/
```

**Label remapping (applied automatically before training):**

| Original Class | Original Value | Remapped Value | Remapped Class |
|----------------|---------------|----------------|----------------|
| Background     | 0             | 0              | No-Change      |
| Intact         | 1             | 0              | No-Change      |
| Damaged        | 2             | 1              | Change         |
| Destroyed      | 3             | 1              | Change         |

---

## Training

```bash
python train.py --config config.yaml
```

All hyperparameters are controlled via `config.yaml`:

```yaml
batch_size: 8
encoder: resnet34
epochs: 30
focal_gamma: 2.0
focal_w: 0.5
dice_w: 0.5
image_size: 256
in_channels: 4
lr: 0.0001
max_samples: 1000
num_workers: 2
seed: 42
threshold: 0.5
weight_decay: 0.0001
```

---

## Evaluation

```bash
python eval.py --data_path /path/to/test --weights /path/to/best_model.pth
```

This will print IoU, Precision, Recall, and F1 for the Change class, plot the confusion matrix, and save qualitative prediction visualisations.

---

## Model Weights

Download the trained model checkpoint:

**[best_model.pth — Google Drive](YOUR_PUBLIC_DRIVE_LINK_HERE)**

Place it at `checkpoints/best_model.pth` before running evaluation.

> Replace `YOUR_PUBLIC_DRIVE_LINK_HERE` with your actual public Google Drive or HuggingFace Hub link.

---

## Results

Metrics computed on the Change class (label = 1):

| Metric    | Validation | Test   |
|-----------|-----------|--------|
| IoU       | 0.3660    | 0.0637 |
| Precision | 0.5162    | 0.1079 |
| Recall    | 0.5570    | 0.1344 |
| F1 Score  | 0.5359    | 0.1197 |

**Best epoch:** 25 / 30

The gap between validation and test performance is primarily attributed to training on a 1,000-sample subset of the full training set due to compute constraints (Colab free tier, T4 GPU). See the technical report for full analysis and proposed remedies.

---

## Citation / References

```
@inproceedings{chen2021bit,
  title={Remote Sensing Image Change Detection with Transformers},
  author={Chen, Hao and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021}
}

@article{fang2023changer,
  title={Changer: Feature Interaction is What You Need for Change Detection},
  author={Fang, Sheng and Li, Kaiyu and Li, Zhe},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2023}
}

@inproceedings{ronneberger2015unet,
  title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={MICCAI},
  year={2015}
}
```

**Libraries:**
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch)
- [albumentations](https://albumentations.ai/)
- [rasterio](https://rasterio.readthedocs.io/)
