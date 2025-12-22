from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

if 'get_ipython' not in globals():
    from config import *
    from dataset_formatter import dataset

# モデルとトークナイザーのロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None, ## None:最適なデータ型を自動選択
    load_in_4bit=True, ## 4bit量子化
)

# PEFT(LoRAの適用)
model = FastLanguageModel.get_peft_model(
    model,
    r=16, ## LoRAのランク：追加するパラメータの次元数 = 2 × r × 元のモデルの次元数
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", ## LoRAを適用するtransformerの層
        "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16, ## LoRAの層の出力をどの程度強く反映させるかを制御：lora_alpha / r 
    bias="none", ## バイアス項は調整しない
)

# データセットをトークナイズ
def dataset_tokenize(example):
    """データセットのテキストをトークナイズする関数"""
    print("tokenize:", example["text"])
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
    )
tokenized_dataset = dataset.map(dataset_tokenize,
    remove_columns=dataset.column_names)

# Fine-Tuningを実行する
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    args=TrainingArguments(
        output_dir=MODEL_SAVE_DIR,
        max_steps=MAX_STEPS, # 最大ステップ数
        per_device_train_batch_size=2, ## バッチサイズ
        gradient_accumulation_steps=4, # 勾配蓄積ステップ数
        warmup_steps=5, # ウォームアップステップ数
        learning_rate=2e-4, # 学習率
        logging_steps=1, # ログ出力ステップ数
        optim="adamw_8bit", # 8bit AdamWを使用
        fp16 = not torch.cuda.is_bf16_supported(), # fp16使用の有無
        bf16 = torch.cuda.is_bf16_supported(), # bf16使用の有無
        report_to="none" # ログを出力しない
    ),
)
trainer.train()

# Fine-Tuningの結果を保存
model.save_pretrained(MODEL_SAVE_DIR)