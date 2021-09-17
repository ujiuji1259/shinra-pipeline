python predict_for_leader_board.py \
    --model_name cl-tohoku/bert-base-japanese \
    --model_path /home/is/ujiie/wiki_en/models/bert_biencoder_without_context.model \
    --output_path /home/is/ujiie/wiki_en/results/sample_wc \
    --cross_model_path /home/is/ujiie/wiki_en/models/bert_crossencoder_negative5_batch8_without_context.model \
    --index_path /home/is/ujiie/wiki_en/models/bert_base_without_context_index \
    --load_index \
    --candidate_dataset /data1/ujiie/wiki_resource/pages_preprocessed_for_bert-base-japanese.pkl \
    --candidate_preprocessed \
    --mention_dataset /data1/ujiie/shinra/EN/linkjp-sample-210402 \
    --category Airport \
    --builder_gpu \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 128 \
    --fp16 \
    --fp16_opt_level O1 \

