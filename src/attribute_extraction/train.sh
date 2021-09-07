DATA_PATH=/data
MODEL_PATH=/models

CUDA_VISIBLE_DEVICES=0,1 python train.py \
    --bert_name cl-tohoku/bert-base-japanese \
    --input_path ${DATA_PATH}/Event/Event_Other \
    --data_split ${DATA_PATH}/data_split/Event_Other \
    --model_path $MODEL_PATH/ \
    --parallel \
    --lr 1e-5 \
    --bsz 32 \
    --epoch 1 \
    --grad_acc 1 \
    --warmup 0.1 \
    --grad_clip 1.0 \
    --seed 42 \
    --note with_two_output_layer \
