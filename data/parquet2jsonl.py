import pandas as pd
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

df = pd.read_parquet(os.path.join(current_dir, "raw_data/train.parquet"))

print(f"Saving train.json to {os.path.join(current_dir, 'raw_data/train.jsonl')}")
df.to_json(os.path.join(current_dir, "raw_data/train.jsonl"), orient='records', lines=True)

# 取前10行数据
df_small = df.head(10)

# 保存到 train_small.json
print(f"Saving train_small.json to {os.path.join(current_dir, 'raw_data/train_small.jsonl')}")
df_small.to_json(os.path.join(current_dir, "raw_data/train_small.jsonl"), orient='records', lines=True)