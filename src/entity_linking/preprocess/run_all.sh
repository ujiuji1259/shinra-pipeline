INPUT_PATH=/data1/ujiie/shinra/EN/linkjp-leader-board-210421
OUTPUT_PATH=/home/is/ujiie/linkjp_scorer/lb_baseline
TITLE2ID=/data1/ujiie/wiki_resource/title2page.csv

categories=("Airport" "City" "Company" "Compound" "Conference" "Lake" "Person")
for v in "${categories[@]}"
do
    python baseline.py \
        --input_path $INPUT_PATH \
        --category $v \
        --title2id $TITLE2ID \
        --output_path ${OUTPUT_PATH}/${v}.json
done