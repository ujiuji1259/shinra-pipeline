DATASET_PATH_PREFIX=/data/training_data_preprocessd_for_bert-base-japanese_9M
CAND_DATASET=/data/pages_preprocessed_for_bert-base-japanese.pkl
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
    --candidate_preprocessed \
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

