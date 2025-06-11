# 写真構図推定

このリポジトリは、写真の構図カテゴリを深層学習で判定するサンプル実装です。PyTorch を用いており、学習済みモデルの作成から評価・推論までを一通り実行できます。

## ディレクトリ構成

```
photo_composition/
  dataset.py   - データセット読み込みユーティリティ
  model.py     - ResNet18 ベースの `CompositionNet`
  train.py     - 学習スクリプト
  predict.py   - 推論スクリプト
  evaluate.py  - テストデータでの評価スクリプト
```

## データセットの準備

トレーニングと検証用のフォルダを以下のように配置します。各クラスフォルダには対応するサリエンシーマップを格納した `saliency/` フォルダを作成し、画像と同名（拡張子は `.pickle`）のファイルを置きます。

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
      image3.jpg
      ...
      saliency/
        image3.pickle
        ...
```

## 依存関係

Python 3.8 以降と以下のライブラリを使用します。

- PyTorch
- torchvision
- scikit-learn (評価レポート用)

その他のライブラリは `requirements.txt` などを参照してください。

## 学習

```bash
python photo_composition/train.py --data-dir <dataset_root> --epochs 10
```

学習後、最も精度の高いモデルが `composition_model.pth` として保存され、クラス名一覧が `composition_model_classes.json` に記録されます。

## 推論

```bash
python photo_composition/predict.py --model composition_model.pth \
    --image path/to/photo.jpg --saliency path/to/photo_saliency.pickle
```
By default the script reads class names from `composition_model_classes.json`.
Use `--class-names` to override them.

`--class-names` を指定しない場合は、モデルファイル名に対応する `<model>_classes.json` が読み込まれます。RGB 画像とサリエンシーマップを組み合わせた 4 チャネル入力で推論を行います。

## 評価

```bash
python photo_composition/evaluate.py --model composition_model.pth \
    --data-dir path/to/test_dataset
```

テストデータ一式を与えると、各クラスごとの精度と全体の精度を表示します。こちらも `--class-names` でクラス名を上書きできます。

## モデルのカスタマイズ

`model.py` の `CompositionNet` は ResNet18 をベースにしています。必要に応じて層構成やハイパーパラメータを変更してみてください。

=======
This prints the classification accuracy on the provided dataset. When
`--class-names` is omitted, the script loads class names from
`composition_model_classes.json`.
