python train.py \
    --bert_name cl-tohoku/bert-base-japanese \
    --input_path /data1/ujiie/shinra/tohoku_bert/Event/Event_Other \
    --data_split /data1/ujiie/shinra/tohoku_bert/data_split/Event_Other \
    --model_path /home/is/ujiie/shinra-pipeline/models/ \
    --lr 1e-5 \
    --bsz 32 \
    --epoch 50 \
    --grad_acc 1 \
    --warmup 0.1 \
    --grad_clip 1.0 \
    --seed 42 \
    --note with_two_output_layer \
