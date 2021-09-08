CUDA_VISIBLE_DEVICES=0,1 python train_bert_biencoder.py \
    --model_name cl-tohoku/bert-base-japanese \
    --mention_dataset /data1/ujiie/wiki_resource/training_data_preprocessd_for_bert-base-japanese_9M.jsonl \
    --mention_index /data1/ujiie/wiki_resource/training_data_preprocessd_for_bert-base-japanese_9M_index.npy \
    --candidate_dataset /data1/ujiie/wiki_resource/pages_preprocessed_for_bert-base-japanese.pkl \
    --hard_negative \
    --num_negs 2 \
    --path_for_NN /data1/ujiie/wiki_resource/data_with_negatives \
    --model_path /home/is/ujiie/wiki_en/models/bert_biencoder_with_negatives_4 \
    --candidate_preprocessed \
    --mention_preprocessed \
    --lr 1e-5 \
    --bsz 32 \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 50 \
    --traindata_size 9000000 \
    --mlflow \
    --gradient_accumulation_steps 1 \
    --max_grad_norm 1.0 \
    --epochs 4 \
    --fp16 \
    --fp16_opt_level O1 \
    --parallel \
    --seed 42

