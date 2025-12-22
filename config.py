MODEL_NAME = "unsloth/mistral-7b-bnb-4bit"
DATASET_NAME = "shi3z/alpaca_cleaned_ja_json"

MAX_DATASET_SIZE = 500 ## データセットの最大サイズ
MAX_SEQ_LENGTH = 2048 ## シーケンス長
MAX_STEPS = 1 ## 学習ステップ数

# モデルの保存先ディレクトリ
MODEL_SAVE_DIR="./rongocho_llm"