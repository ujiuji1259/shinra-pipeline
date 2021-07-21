python train.py \
    --bert_name cl-tohoku/bert-base-japanese \
    --input_path /data1/ujiie/shinra/tohoku_bert/Event/Event_Other \
    --attribute_list /data1/ujiie/shinra/tohoku_bert/attributes.pickle \
    --data_split /data1/ujiie/shinra/tohoku_bert/data_split/Event_Other \
    --lr 1e-5 \
    --bsz 32 \
    --epoch 50 \
    --grad_acc 1 \
    --warmup 0.1 \
    --grad_clip 1.0 \
    --note with_scheduler_and_with_gradient_clip
