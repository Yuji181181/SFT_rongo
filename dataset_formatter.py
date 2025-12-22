from datasets import load_dataset
if 'get_ipython' not in globals():
    from config import *

# データセットを読み込んで最大件数だけ取り出す
dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.select(range(MAX_DATASET_SIZE))

# プロンプトテンプレートの指定
PROMPT_TEMPLATE = """\
<s>あなたは偉い先生である。次のタスクに対する指示文に対して内容に基づいて論語風に答えよ。

### 指示:
{}{}

### 応答:
子曰く、{}</s>
"""

# データセットを整形する関数を用意
def format_example(example):
    """テキストを整形する関数"""
    # データセットから値を取り出す
    instruction = example["instruction"].strip()
    input = example["input"].strip()
    output = example["output"].strip()
    # 応答を「である調」に変換する
    output = output.replace("です。", "である。")
    output = output.replace("ちます。", "つ。")
    output = output.replace("でした", "だった")
    output = output.replace("できます", "できる")
    output = output.replace("します", "するのだ")
    output = output.replace("あります。", "あるのだ。")
    output = output.replace("してください", "してみよ")
    # プロンプトテンプレートに埋め込む
    text = PROMPT_TEMPLATE.format(instruction, input, output)
    print("text=", text)
    return {"text": text}

# データセットの整形を適用
dataset = dataset.map(format_example)