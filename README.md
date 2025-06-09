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

## 環境構築

1. Python 3.8 以上がインストールされていることを確認します。
2. 任意で仮想環境を作成して有効化します。

```bash
python3 -m venv venv
source venv/bin/activate
```

3. 依存パッケージをインストールします。GPU を使用する場合は PyTorch の公式サイトで案内されているコマンドに読み替えてください。

```bash
pip install torch torchvision pillow
```

## 使い方

1. 以下の構造で学習用データセットを用意します。

```
<dataset_root>/
  train/
    <class_name>/image1.jpg
    ...
  val/
    <class_name>/image2.jpg
    ...
```

2. 学習を実行します。`<dataset_root>` には上記のディレクトリを指定します。

```bash
python photo_composition/train.py --data-dir <dataset_root> --epochs 10
```

3. 学習済みモデルで画像の構図を推定します。

```bash
python photo_composition/predict.py --model composition_model.pth \
    --class-names rule_of_thirds,centered,diagonal \
    --image path/to/photo.jpg
```

