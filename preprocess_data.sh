# 检查是否已经解压，如果不存在则解压
if [ ! -f "data/raw_data/train.parquet" ]; then
    echo "解压 train.parquet 文件..."
    rar x data/raw_data/train.parquet.part1.rar data/raw_data/
else
    echo "train.parquet 文件已存在，跳过解压"
fi

python data/parquet2json.py

mkdir -p data/preprocessed_data

python tools/preprocess_data.py \
    --input data/raw_data/train.jsonl \
    --output-prefix data/preprocessed_data/train \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model tokenizer/Mixtral-8x7B \
    --json-keys content \
    --workers 16 \