# shinra-pipeline

## 環境構築
### 依存パッケージ
- [apex](https://github.com/NVIDIA/apex)
- pytorch
- transformers
- [faiss](https://github.com/facebookresearch/faiss)
- mlflow
- seqeval
- fugashi
- ipadic

### Docker
`docker build .`

## 属性抽出
### データの準備
1. SHINRA2020での[前処理済みデータ](http://shinra-project.info/shinra2020jp/data_download/)の森羅2020-JPタスクの学習・ターゲットデータ（トークナイズ済み, Mecab(IPA辞書)&BPE使用, 東北大BERT対応)をダウンロード
2. 学習・検証のデータ分割、ラベル情報をダウンロード[データ](https://drive.google.com/drive/folders/1WfzIut-f4ka_5ToyTtVRbtff92tQst63?usp=sharing)
3. 以下のようにデータフォルダを構成
```bash
data
|-Event
  |-Event_Other
  ...
|-
...
|-data_split
|-attributes
```

### モデルのダウンロード
学習済みモデルのダウンロード

### 予測
1. 出力用フォルダを作成（`shinra-pipeline/outputs`など）
2. `predict_all_categories.sh`の`DATA_PATH`，`MODEL_PATH`，`OUTPUT_PATH`を変更
3. `bash predict_all_categories.sh`
