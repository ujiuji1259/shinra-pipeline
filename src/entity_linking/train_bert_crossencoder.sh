DATASET_PATH_PREFIX=/data/training_data_preprocessd_for_bert-base-japanese_1M_NNs
CAND_DATASET=/data/pages_preprocessed_for_bert-base-japanese.pkl
MODEL_PATH=/models/bert_crossencoder_with_negative_fixed.model


CUDA_VISIBLE_DEVICES=0,1 python train_bert_crossencoder.py \
    --model_name cl-tohoku/bert-base-japanese \
    --mention_dataset ${DATASET_PATH_PREFIX}.jsonl \
    --mention_index ${DATASET_PATH_PREFIX}_index.npy \
    --candidate_dataset ${CAND_DATASET} \
    --model_path ${MODEL_PATH} \
    --candidate_preprocessed \
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

