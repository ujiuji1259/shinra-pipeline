DATA_PATH=/data
MODEL_PATH=/models
OUTPUT_PATH=/workspace/outputs/sample

all_categories=("Airport" "City" "Company" "Compound" "Conference" "Lake" "Person")

for CATEGORY in ${all_categories[@]}; do
    python predict.py \
        --model_name cl-tohoku/bert-base-japanese \
        --biencoder_path ${MODEL_PATH}/bert_biencoder_with_negatives_4_1.model \
        --output_path ${OUTPUT_PATH}/${CATEGORY}.json \
        --crossencoder_path ${MODEL_PATH}/bert_crossencoder_negative5_batch8_v2.model \
        --index_path ${MODEL_PATH}/bert_base_with_negatives \
	--load_index \
        --candidate_dataset ${DATA_PATH}/pages_preprocessed_for_bert-base-japanese.pkl \
	--candidate_preprocessed \
        --input_path ${DATA_PATH}/linkjp-sample-210402 \
        --category ${CATEGORY} \
        --builder_gpu \
        --faiss_gpu_id 0 \
        --max_ctxt_len 32 \
        --max_title_len -1 \
        --max_desc_len 128 \
        --fp16 \
        --fp16_opt_level O1 
done
