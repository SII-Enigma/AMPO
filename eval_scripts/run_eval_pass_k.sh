#!/bin/bash

ROOT='Your ROOT PATH'

cd $ROOT/AMPO/eval_scripts

DATA=$ROOT/AMPO/data/valid_pass_at_256.parquet
OUTPUT_DIR=$ROOT/AMPO/results/pass_k
k=256
interval=16

MODEL_PATHS=(
    "/path/to/Qwen2.5-7B-Ins-AMPO"
)
MODEL_NAMES=(
    "Qwen2.5-7B-Ins-AMPO"
)
TEMPLATES=(
    "own"
)

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH=${MODEL_PATHS[$i]}
    MODEL_NAME=${MODEL_NAMES[$i]}
    TEMPLATE=${TEMPLATES[$i]}

    echo "Running inference for $MODEL_NAME ..."

    python eval_pass_k.py \
      --model_path "$MODEL_PATH" \
      --input_file "$DATA" \
      --remove_system True \
      --output_file "$OUTPUT_DIR/$MODEL_NAME.jsonl" \
      --template "$TEMPLATE" \
      --n $k\
      --m $interval
done

