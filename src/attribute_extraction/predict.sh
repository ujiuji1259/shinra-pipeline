CUDA_VISIBLE_DEVICES=0,1
python train.py \
    --bert_name cl-tohoku/bert-base-japanese \
    --input_path /data1/ujiie/shinra/tohoku_bert/Event/Event_Other \
    --model_path /home/is/ujiie/shinra-pipeline/models/Event_Other.model \
    --output_path /home/is/ujiie/shinra-pipeline/outputs/Event_Other.jsonl \
    --bsz 64 \
    --note with_two_output_layer \
