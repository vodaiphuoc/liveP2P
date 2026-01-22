#!/bin/bash

OPTS=$(getopt -o "" --long merged_output_path: -- "$@")
eval set -- "$OPTS"

git clone https://github.com/ggerganov/llama.cpp

echo "install dependecies"
cd llama.cpp && pip install -r requirements.txt

echo "Building llama.cpp"
make -j$(nproc)

python ./llama.cpp/convert_hf_to_gguf.py "$2" --outfile model-fp16.gguf --outtype f16 && \
./llama.cpp/quantize model-fp16.gguf model-q4.gguf Q4_K_M
