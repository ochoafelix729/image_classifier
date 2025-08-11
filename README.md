# Image Classifier with ResNet‑50

An **image classification pipeline** powered by a pre‑trained **ResNet‑50** from `torchvision`. The notebook takes input images, applies the standard ImageNet preprocessing, and returns the **top prediction with confidence** (and optionally **Top‑K**).

---

## Features
- **Pretrained backbones** – `torchvision.models.resnet50(pretrained=True)` (ImageNet‑1k).
- **Correct preprocessing** – resize → center‑crop → tensor → normalize (ImageNet mean/std).
- **Top‑K** predictions – inspect the K most likely classes.
- **Batch mode** – classify multiple images in one run.
- **Inline visualization** – show the image with prediction + confidence in the notebook.

---

## Project Structure
```
.
├── image_classifier.ipynb       # Main notebook
├── imagenet_class_names.pkl     # (Optional) Index → label mapping (ImageNet-1k)
├── dog.jpg                      # Example image (optional)
├── image1.jpeg … image5.jpeg    # Sample images (optional)
└── README.md
```

> **Note**: If `imagenet_class_names.pkl` is not provided, you can map indices to class names via other sources or disable the pretty‑name lookup.

---

## Installation

1) **Clone**:
```bash
git clone https://github.com/ochoafelix729/image_classifier.git
cd image-classifier
```

2) **Python dependencies**:
```bash
pip install torch torchvision pillow matplotlib
```

---

## Notebook Workflow
1. **Load** pretrained ResNet‑50 and set `eval()`.
2. **Transform** each image using ImageNet normalization.
3. **Infer** with softmax to get probabilities.
4. **Display** the input alongside its predicted label and confidence.