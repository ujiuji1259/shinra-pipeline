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
[こちら](src/attribute_extraction/README.md)を参照してください．

## Entity Linking
[こちら](src/entity_linking/README.md)を参照してください．

## Pipeline
`{page_id}.txt`の形式の同一カテゴリのwikipedia記事を入力として、属性抽出、entity linkingを行い、森羅の提出フォーマットで結果を出力します．

### 各種データ
- 入力データ
```
input_dir
|-- 1001711.txt
|-- 1002892.txt
|-- 1039549.txt
|-- 1042592.txt
|-- 106524.txt
|-- 111707.txt
...
```

- 各カテゴリの属性
```
attributes
|-- Accommodation.txt
|-- Airport.txt
|-- Amusement_Park.txt
|-- Archaeological_Place_Other.txt
|-- Bay.txt
|-- Bridge.txt
|-- Canal.txt
|-- Car_Stop.txt
...
```

その他のデータについては[属性抽出](src/attribute_extraction/README.md)，[entity_linking]((src/entity_linking/README.md))のREADMEを参照ください．

### 推論
```
cd src
bash pipeline.sh
```

```
python pipeline.py \
    --input_path /path/to/wiki_dir \
    --category Airport \
    --output_path /path/to/output.json \
    --ner_path /path/to/ner_model \
    --attribute_path /path/to/attribute_dir \
    --model_name cl-tohoku/bert-base-japanese \
    --biencoder_path biencoder.model \
    --crossencoder_path crossencoder.model \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 128 \
    --candidate_dataset /path/to/candidates.jsonl \
    --index_path /path/to/faiss_candidates \
    --builder_gpu \
    --faiss_gpu_id 0 \
    --parallel \
    --fp16 \
    --fp16_opt_level O
```

- input_path: 記事のディレクトリ
- attribute_path: `{category}.txt`形式の1行1属性となるファイルのディレクトリ