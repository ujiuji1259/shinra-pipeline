all_categories=("Event" "Facility" "JP-5" "Location" "Organization")
ATTRIBUTE_LIST="/data1/ujiie/shinra/tohoku_bert/attributes.pickle"
DATA_SPLIT="/data1/ujiie/shinra/tohoku_bert/data_split/"
MODEL_PATH="/home/is/ujiie/shinra-pipeline/models/"
for fn in /data1/ujiie/shinra/tohoku_bert/*/*
do
    cat=`dirname $fn | sed 's/.*\/\([^/.]*\)$/\1/'`
    if printf '%s\n' "${all_categories[@]}" | grep -qx $cat > /dev/null >&2; then
        category=`basename $fn`
        python train.py \
            --bert_name cl-tohoku/bert-base-japanese \
            --input_path $fn \
            --attribute_list $ATTRIBUTE_LIST \
            --data_split $DATA_SPLIT$category \
            --model_path "${MODEL_PATH}${category}.model" \
            --lr 1e-5 \
            --bsz 32 \
            --epoch 50 \
            --grad_acc 1 \
            --warmup 0.1 \
            --grad_clip 1.0 \
            --note "${category} with 2 output layers without warmupstep fix"
    fi
done