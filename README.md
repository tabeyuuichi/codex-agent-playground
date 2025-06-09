# Photo Composition Estimation

This repository contains an example implementation of photo composition estimation using deep learning.

The code is written with PyTorch and provides utilities for training a classification model that predicts the composition category of a given photograph (e.g., rule of thirds, centered, diagonal composition).

## Directory Structure

```
photo_composition/
  dataset.py   - utilities for loading datasets using `torchvision.datasets.ImageFolder`
  model.py     - `CompositionNet` CNN model based on ResNet18
  train.py     - training script
  predict.py   - inference script
```

## Usage

1. Prepare a dataset with the following structure:

```
<dataset_root>/
  train/
    <class_name>/image1.jpg
    ...
  val/
    <class_name>/image2.jpg
    ...
```

2. Install dependencies (PyTorch and torchvision are required).
3. Train the model:

```
python photo_composition/train.py --data-dir <dataset_root> --epochs 10
```

4. Predict composition for a new image:

```
python photo_composition/predict.py --model composition_model.pth \
    --class-names rule_of_thirds,centered,diagonal \
    --image path/to/photo.jpg
```
