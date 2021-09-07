DATA_PATH=/data
MODEL_PATH=/models
OUTPUT_PATH=/workspace/outputs

# all_categories=("Event" "Facility" "JP-5" "Location" "Organization")
all_categories=("Event")
for fn in ${DATA_PATH}/*/*
do
    cat=`dirname $fn | sed 's/.*\/\([^/.]*\)$/\1/'`
    if printf '%s\n' "${all_categories[@]}" | grep -qx $cat > /dev/null >&2; then
        category=`basename $fn`
        CUDA_VISIBLE_DEVICES=0,1 python predict.py \
            --bert_name cl-tohoku/bert-base-japanese \
            --input_path $fn \
            --model_path $MODEL_PATH/${category}_best.model \
            --output_path $OUTPUT_PATH/${category}.jsonl \
            --bsz 32 \
            --fp16 \
            --fp16_opt_level O1 \
            --note with_two_output_layer
    fi
done