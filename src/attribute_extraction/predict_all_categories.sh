DATA_PATH=/data1/ujiie/shinra/tohoku_bert2
MODEL_PATH=/data1/ujiie/model_for_shinra
OUTPUT_PATH=/home/is/ujiie/shinra-pipeline/outputs

# all_categories=("Event" "Facility" "JP-5" "Location" "Organization")
all_categories=("Organization")
for fn in ${DATA_PATH}/*/*
do
    cat=`dirname $fn | sed 's/.*\/\([^/.]*\)$/\1/'`
    if printf '%s\n' "${all_categories[@]}" | grep -qx $cat > /dev/null >&2; then
        category=`basename $fn`
        CUDA_VISIBLE_DEVICES=2 python predict.py \
            --bert_name cl-tohoku/bert-base-japanese \
            --input_path $fn \
            --model_path $MODEL_PATH/${category}_best.model \
            --output_path $OUTPUT_PATH/${category}.jsonl \
            --bsz 16 \
            --fp16 \
            --fp16_opt_level O1 \
            --note with_two_output_layer
    fi
done
