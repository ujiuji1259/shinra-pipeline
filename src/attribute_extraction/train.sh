CUDA_VISIBLE_DEVICES=0,1
python train.py \
    --bert_name cl-tohoku/bert-base-japanese \
    --input_path /home/is/ujiie/shinra/Event/Event_Other \
    --data_split /home/is/ujiie/shinra/data_split/Event_Other \
    --model_path /home/is/ujiie/shinra-pipeline/models/ \
    --parallel \
    --lr 1e-5 \
    --bsz 32 \
    --epoch 1 \
    --grad_acc 1 \
    --warmup 0.1 \
    --grad_clip 1.0 \
    --seed 42 \
    --note with_two_output_layer \
