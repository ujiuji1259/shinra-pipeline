# all_categories=("Event" "Facility" "JP-5" "Location" "Organization")
all_categories=("JP-5")
DATA_SPLIT="${DATA_PATH}/data_split/"
for fn in ${DATA_PATH}/*/*
do
    cat=`dirname $fn | sed 's/.*\/\([^/.]*\)$/\1/'`
    if printf '%s\n' "${all_categories[@]}" | grep -qx $cat > /dev/null >&2; then
        category=`basename $fn`
        python train.py \
            --bert_name cl-tohoku/bert-base-japanese \
            --input_path $fn \
            --data_split $DATA_SPLIT$category \
            --model_path "${MODEL_PATH}" \
            --lr 1e-5 \
            --bsz 16 \
            --epoch 50 \
            --grad_acc 1 \
            --warmup 0.1 \
            --grad_clip 1.0 \
            --seed 42 \
            --note "${category} with 2 output layers without warmupstep fix"
    fi
done
