DATASET_PATH=/data
MODEL_PATH=/models

CUDA_VISIBLE_DEVICES=0 python save_nearest_neighbor.py \
    --model_name cl-tohoku/bert-base-japanese \
    --model_path ${MODEL_PATH}/bert_biencoder_with_negatives_4_1.model \
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


