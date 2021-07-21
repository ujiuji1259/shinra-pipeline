python pipeline.py \
    --input /data1/ujiie/shinra/tohoku_bert/Event/Event_Other \
    --attribute /data1/ujiie/shinra/tohoku_bert/attributes.pickle \
    --model_name cl-tohoku/bert-base-japanese \
    --generator_path /home/is/ujiie/wiki_en/models/bert_biencoder.model \
    --ranker_path /home/is/ujiie/wiki_en/models/bert_crossencoder_negative5_batch8_v2.model \
    --index_path /home/is/ujiie/wiki_en/models/base_bert_index \
    --load_index \
    --candidate_dataset /data1/ujiie/wiki_resource/pages_preprocessed_for_bert-base-japanese.pkl \
    --candidate_preprocessed \
    --builder_gpu \
    --max_ctxt_len 32 \
    --max_title_len -1 \
    --max_desc_len 200 \
    --fp16 \
    --fp16_opt_level O1 \
    --parallel \

