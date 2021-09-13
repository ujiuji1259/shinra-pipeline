export CUDA_VISIBLE_DEVICES=1,0

python train_bert_crossencoder.py \
    --model_name cl-tohoku/bert-base-japanese \
    --mention_dataset /data1/ujiie/wiki_resource/training_data_preprocessed_for_bert-base-japanese-NN100_context64.jsonl \
    --mention_index /data1/ujiie/wiki_resource/training_data_preprocessed_for_bert-base-japanese-NN100_context64_index.npy \
    --candidate_dataset /data1/ujiie/wiki_resource/pages_preprocessed_for_bert-base-japanese.pkl \
    --model_path /home/is/ujiie/wiki_en/models/bert_crossencoder_negative5_batch8_context64.model \
    --candidate_preprocessed \
    --mention_preprocessed \
    --lr 1e-5 \
    --negatives 5 \
    --batch_size 8 \
    --max_ctxt_len 64 \
    --max_title_len -1 \
    --max_desc_len 128 \
    --traindata_size 1000000 \
    --mlflow \
    --model_save_interval 10000 \
    --grad_acc_step 1 \
    --max_grad_norm 1.0 \
    --epochs 4 \
    --fp16 \
    --fp16_opt_level O1 \
    --parallel \
    --logging

