# 写真構図推定リポジトリの使い方

このリポジトリは、RGB画像と対応するサリエンシーマップを組み合わせて写真の構図カテゴリを推定する PyTorch 製プロジェクトです。`photo_composition` ディレクトリに学習・推論・評価・ユーティリティがまとまっており、カスタムデータセットでのトレーニングから学習済みモデルの解析までを一貫して行えます。

## 主なスクリプトと機能

| スクリプト | 用途 |
| --- | --- |
| `train.py` | 学習・検証ループの実行および最良モデルの保存 |
| `predict.py` | 単一画像に対する推論と結果の表示 |
| `evaluate.py` | テストデータに対する精度評価と分類レポートの出力 |
| `dataset.py` | RGB+サリエンシー4チャネル入力用のデータローダ定義 |
| `model.py` | ResNet18 を拡張した `CompositionNet` のモデル定義 |
| `show_weights.py` | 学習済み重みを読み込んで全パラメータを表示する CLI ユーティリティ |

## 必要環境

- Python 3.8 以上
- PyTorch / torchvision
- scikit-learn（`classification_report` を使用）
- Pillow, numpy などの補助ライブラリ

プロジェクト直下に `requirements.txt` がある場合は `pip install -r requirements.txt` でまとめて導入してください。

## データセットの構造

サリエンシーマップ（`.pickle`）を画像と同名ファイルとして `saliency/` フォルダに配置します。学習・検証・テストで同じ構造を使用します。

```
<dataset_root>/
  train/
    <class_name>/
      image1.jpg
      image2.jpg
      ...
      saliency/
        image1.pickle
        image2.pickle
        ...
  val/
    <class_name>/
      ...
      saliency/
        ...
  test/
    <class_name>/
      ...
      saliency/
        ...
```

サリエンシーマップの値域は学習時と推論時で一致させてください（例: 0〜255 の `uint8` か 0〜1 の `float`）。

## 学習の実行

```bash
python photo_composition/train.py \
    --data-dir <dataset_root> \
    --epochs 20 \
    --batch-size 32 \
    --lr 1e-4
```

- 最良エポックの重みが `composition_model.pth` として保存されます。
- 同時にクラス名のリストが `composition_model_classes.json` に出力されます。
- モデルは RGB+サリエンシーの4チャネル入力を受け付けるよう、ResNet18 の1層目を拡張しています。

## 推論

単一画像で推論する際は RGB 画像とサリエンシーマップを指定します。

```bash
python photo_composition/predict.py \
    --model composition_model.pth \
    --image path/to/image.jpg \
    --saliency path/to/image_saliency.pickle
```

- `--class-names` を指定しない場合、モデルファイル名に対応する `<model>_classes.json` を自動で読み込みます。
- 出力にはクラスごとのスコアと予測ラベルが含まれます。

## 評価

テストデータ全体の性能を測定します。

```bash
python photo_composition/evaluate.py \
    --model composition_model.pth \
    --data-dir path/to/test_dataset
```

- 精度・適合率・再現率などの指標が表示されます。
- `--class-names` でクラス一覧を外部ファイルから読み込むことができます。

## 重みの確認

学習済みモデルの各パラメータを確認するには `show_weights.py` を使用します。

```bash
python photo_composition/show_weights.py \
    --model composition_model.pth \
    --classes composition_model_classes.json
```

ファイルパスを省略した場合は同ディレクトリ内の既定ファイルを参照します。大規模モデルの場合は出力が膨大になるため、`grep` やリダイレクトでの絞り込みを推奨します。

## カスタマイズのヒント

- `model.py` の `CompositionNet` を変更して ResNet の層構成やヘッドを差し替えられます。
- `dataset.py` の前処理（リサイズ、正規化、サリエンシーマップのスケーリングなど）を調整することで性能に影響を与えられます。
- `train.py` 内のハイパーパラメータ（学習率、オプティマイザ、スケジューラ）も適宜編集してください。

## ライセンス

本リポジトリは `LICENSE` ファイルに記載された条件に従います。
