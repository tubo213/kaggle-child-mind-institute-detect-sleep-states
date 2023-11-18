#!/bin/bash

current_dir=$(pwd)

conf_content="
data_dir: $current_dir/data
processed_dir: $current_dir/processed
output_dir: $current_dir/output
model_dir: $current_dir/output/train
sub_dir: ./
"

echo "$conf_content" > "$current_dir/run/conf/dir/local.yaml"
