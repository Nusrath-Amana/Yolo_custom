# Shape-Aware DIoU (SA-DIoU) Loss

An enhanced bounding box similarity metric combining overlap, center alignment, and aspect ratio awareness. Improves object detection performance for shape-sensitive tasks.

## Key Features

- **Comprehensive Consideration**: Jointly optimizes overlap, center alignment, and aspect ratio
- **Faster Convergence**: Achieves optimal mAP in fewer epochs than CIoU
- **Improved Accuracy**: 4-5% mAP improvement over standard IoU metrics

## Mathematical Formulation

SA-DIoU extends DIoU with a shape consistency term:

```math
\text{SA-DIoU} = \underbrace{\text{IoU} - \frac{\rho^2}{c^2}}_{\text{DIoU}} + \lambda \cdot \underbrace{\left|\frac{w_1}{h_1} - \frac{w_2}{h_2}\right|}_{\text{SCT}}
```

Where:
- **IoU**: Standard Intersection over Union
- **ρ**: Center distance between boxes
- **c**: Diagonal of smallest enclosing box
- **SCT**: Shape Consistency Term (aspect ratio difference)
- **λ**: Weighting hyperparameter (default: 0.05)

## Usage

### Clone Repository
```bash
git clone https://github.com/Nusrath-Amana/Yolo_custom
cd Yolo_custom
```

### Option 1: YOLOv5 with Ultralytics

```bash
cd yolov5
pip install -r requirements.txt

# Train with SA-DIoU loss
python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5m.pt --cache

# Validate model
python val.py --weights best.pt --data data.yaml --img 640
```

### Option 2: Minimal YOLOv5 Implementation

```bash
cd minimal/yolov5

# Train with SA-DIoU loss
python train.py

# Test on CPU
python test.py --device 'cpu'
```

## Performance Results

The SA-DIoU loss function demonstrates significant improvements:
- Faster convergence (requires fewer epochs)
- Higher mAP (+4-5%) compared to standard IoU variants
- Better handling of objects with distinct aspect ratios

## Limitations and Future Work

- The λ parameter may need tuning for specific datasets
- Performance on objects with similar centers and aspect ratios but different sizes could be improved
- Future versions will explore dynamic λ adjustment and size-aware penalties
