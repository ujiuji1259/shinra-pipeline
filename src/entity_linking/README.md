# Entity Linking

## 手法
[Wu L, Petroni F, Josifoski M, Riedel S, Zettlemoyer L. Scalable Zero-shot Entity Linking with Dense Entity Retrieval. In EMNLP, 2020.](https://aclanthology.org/2020.emnlp-main.519/)

## データ
### 大まかな流れ
1. [評価データ（or サンプルデータ）](https://drive.google.com/file/d/1b9Xm-Qd1sVfmDr8o4y3t-dVnGai15P-q/view)のダウンロード
2. Wikipediaの[ダンプデータ](https://drive.google.com/file/d/1qWA93bpb4uvxv0OZ_Gd3NDGu0Jac9W86/view?usp=sharing)をダウンロード
3. [wiki_extractor](https://github.com/attardi/wikiextractor)でダンプデータの前処理（`--json`をつけておく）
4. 紐付け先データの作成
4. 学習データの作成

### 紐付け先データの作成
以下のようなデータを作成します．
```
{'id': '5', 'title': 'アンパサンド', 'description': 'アンパサンド（&, 英語: ampersand）は、並立助詞'}
{'id': ...}
```

```bash
python src/entity_linking/create_candidate_dataset.py \
    --wiki_dir /path/to/wiki_preprocessed_by_wikiextractor \
    --output_path /path/to/output_file
```

### 学習データの作成
以下のようなデータを作成します．
```
{'pre_sent': '...', 'left_context': '...', 'mention': '...', 'right_context': '...', 'post_sent': '...', 'linkpage_id': '...'}
{'pre_sent': '...', 'left_context': '...', 'mention': '...', 'right_context': '...', 'post_sent': '...', 'linkpage_id': '...'}
```

```bash
python src/entity_linking/create_mention_dataset.py \
    --wiki_dir /path/to/wiki_preprocessed_by_wikiextractor \
    --output_path /path/to/output_file \
    --index_path /path/to/index.npy \
    --preprocess \
    --bert_name cl-tohoku/bert-base-japanese
```

- index_path: データセットの逐次読み込みのための各行の開始位置を`npy`形式で保存します
- preprocess: mentionなどをあらかじめtokenizeしておきます
- bert_name: tokenizerの種類

biencoderの学習にはこのうちの9M，crossencoderには1Mを使用します．
`shuf -n /path/to/jsonl > /path/to/output`とかで9Mをサンプリングしておきます．
なお，サンプリング後のデータの`index`は`create_mention_index.py`を使用して作り直せます．

## Bi-encoder
`bash train_bert_biencoder_hard_negative.sh`

```bash
DATASET_PATH_PREFIX=/data/training_data_preprocessd_for_bert-base-japanese_9M
CAND_DATASET=/data/pages_preprocessed_for_bert-base-japanese.jsonl
NEG_PATH=/data/data_with_negatives
MODEL_PATH_PREFIX=/models/bert_biencoder_with_negatives_4

CUDA_VISIBLE_DEVICES=0,1 python train_bert_biencoder.py \
    --model_name cl-tohoku/bert-base-japanese \
    --mention_dataset ${DATASET_PATH_PREFIX}.jsonl \
    --mention_index ${DATASET_PATH_PREFIX}_index.npy \
    --candidate_dataset ${CAND_DATASET} \
    --hard_negative \
    --num_negs 2 \
    --builder_gpu \
    --faiss_gpu_id 0 \
    --path_for_NN ${NEG_PATH} \
    --model_path ${MODEL_PATH_PREFIX} \
    --mention_preprocessed \
    --lr 1e-5 \
    --bsz 128 \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 128 \
    --traindata_size 9000000 \
    --mlflow \
    --grad_acc_step 1 \
    --max_grad_norm 1.0 \
    --epochs 4 \
    --fp16 \
    --fp16_opt_level O1 \
    --parallel \
    --seed 42
```

- hard_negative: epochの初めに全データについて近傍探索を行い、近傍entityを学習に用います
- num_negs: 使用するnegative sample数．hard_negativeの場合のみ有効
- builder_gpu: 近傍探索にgpuを用います
- faiss_gpu_id: 近傍探索に使用するgpuのidを指定します．
- path_for_NN: 各epochでの近傍探索結果を保存するディレクトリ．hard_negativeの場合のみ有効
- model_path: モデルの保存パス．これをprefixとして`{model_path}_{epoch}.model`で保存

## Cross-encoder
### 前処理
使用する学習データをあらかじめ近傍探索しておきます．
`bash save_nearest_neighbor.sh`

```
DATASET_PATH=/data
MODEL_PATH=/models

CUDA_VISIBLE_DEVICES=0 python save_nearest_neighbor.py \
    --model_name cl-tohoku/bert-base-japanese \
    --model_path ${MODEL_PATH}/mymodel.model \
    --index_path ${MODEL_PATH}/index_negs_1 \
    --output_path ${DATASET_PATH}/training_data_preprocessd_for_bert-base-japanese_1M_NNs.jsonl \
    --index_output_path ${DATASET_PATH}/training_data_preprocessd_for_bert-base-japanese_1M_NNs_index.npy \
    --mention_dataset ${DATASET_PATH}/training_data_preprocessd_for_bert-base-japanese_1M.jsonl \
    --mention_index ${DATASET_PATH}/training_data_preprocessd_for_bert-base-japanese_1M_index.npy \
    --candidate_dataset ${DATASET_PATH}/pages_preprocessed_for_bert-base-japanese.pkl \
    --candidate_preprocessed \
    --mention_preprocessed \
    --builder_gpu \
    --traindata_size 1000000 \
    --NNs 100 \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 128 \
    --batch_size 64 \
    --fp16 \
    --fp16_opt_level O1 \
```

- index_path: 近傍探索モデルの保存用ディレクトリ．


### 学習
`bash train_bert_crossencoder.sh`

```bash
DATASET_PATH_PREFIX=/data/training_data_preprocessd_for_bert-base-japanese_1M_NNs
CAND_DATASET=/data/pages_preprocessed_for_bert-base-japanese.jsonl
MODEL_PATH=/models/bert_crossencoder_with_negative_fixed.model


CUDA_VISIBLE_DEVICES=0,1 python train_bert_crossencoder.py \
    --model_name cl-tohoku/bert-base-japanese \
    --mention_dataset ${DATASET_PATH_PREFIX}.jsonl \
    --mention_index ${DATASET_PATH_PREFIX}_index.npy \
    --candidate_dataset ${CAND_DATASET} \
    --model_path ${MODEL_PATH} \
    --mention_preprocessed \
    --lr 2e-5 \
    --negatives 100 \
    --batch_size 1 \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 128 \
    --traindata_size 1000000 \
    --mlflow \
    --model_save_interval 10000 \
    --grad_acc_step 1 \
    --max_grad_norm 1.0 \
    --epochs 1 \
    --fp16 \
    --fp16_opt_level O1 \
    --parallel \
    --logging \
    --seed 42
```

- negatives: 学習に使用する近傍数

## 推論
### 1カテゴリの予測
`predict.sh`

```
DATA_PATH=/data
MODEL_PATH=/models
OUTPUT_PATH=/workspace/outputs
CATEGORY=Airport

python predict.py \
    --model_name cl-tohoku/bert-base-japanese \
    --biencoder_path ${MODEL_PATH}/mymodel.model \
    --output_path ${OUTPUT_PATH}/${CATEGORY}.json \
    --crossencoder_path ${MODEL_PATH}/bert_crossencoder_with_negative.model \
    --index_path ${MODEL_PATH}/bert_base_without_context_index \
    --load_index \
    --candidate_dataset ${DATA_PATH}/pages_preprocessed_for_bert-base-japanese.jsonl \
    --input_path ${DATA_PATH}/linkjp-sample-210402 \
    --category ${CATEGORY} \
    --builder_gpu \
    --faiss_gpu_id 0 \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 128 \
    --fp16 \
    --fp16_opt_level O1 \
```

- load_index: 近傍探索モデルを保存している場合，それを読み込む

### 全カテゴリの予測
`bash predict_all_categories.sh`