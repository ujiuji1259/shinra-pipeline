DATA_PATH=/data
MODEL_PATH=/models
OUTPUT_PATH=/workspace/shinra-pipeline/outputs

CUDA_VISIBLE_DEVICES=0,1 python predict.py \
    --bert_name cl-tohoku/bert-base-japanese \
    --input_path $DATA_PATH/Event/Event_Other \
    --model_path $MODEL_PATH/Event_Other_best.model \
    --output_path $OUTPUT_PATH/Event_Other.jsonl \
    --bsz 32 \
    --fp16 \
    --fp16_opt_level O1 \
    --note with_two_output_layer \
