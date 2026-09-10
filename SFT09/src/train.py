import os, re
import random
import json
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import load_dataset, DatasetDict
from pathlib import Path
from tqdm.notebook import tqdm
from functools import partial

os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

##事前学習済みのモデルを読み込み##

MODEL_NAME = "unsloth/gemma-2-9b-bnb-4bit"
MAX_SEQ_LENGTH = 2048

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True, 
    dtype=None, 
)
##MMLUデータセットの読み込み##
dataset = load_dataset("cais/mmlu", "all")

##データの中身##
dataset["test"].to_pandas()

##データの成形##
prompt = """{}

### Choices:
{}

### Answer:
{}"""


def formatting_prompts_func(examples, train=True):
    
    questions = examples["question"]
    choice_lists = examples["choices"]
    answers = examples["answer"]

    texts = []
    for question, choices, answer in zip(questions, choice_lists, answers):
        # 評価であればanswerを追加しない
        if not train:
            answer = ""

        # テンプレートに代入してモデル入力を作成
        text = prompt.format(question, choices, answer) # WRITE ME

        # 訓練であれば最後にEOSトークンを追加
        if train:
            # tokenizerをグローバルスコープで使用していることに注意
            text += tokenizer.eos_token # WRITE ME

        texts.append(text)

    return {"text": texts}


# 成形処理の実行
train_dataset = dataset["test"].map(formatting_prompts_func, batched=True) # WRITE ME
test_dataset = dataset["validation"].map(
    partial(formatting_prompts_func, train=False),
    batched=True,
)

##SFT用モデルの準備##
model = FastLanguageModel.get_peft_model(
    base_model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

training_arguments = SFTConfig(
    dataset_text_field="text",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=5,
    num_train_epochs=1,
    learning_rate=2e-5,
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    report_to="tensorboard",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=training_arguments,
)

##学習開始##
trainer.train()

##評価##
def evaluate(dataset, model, tokenizer):
    # モデルを評価モードにする
    model = FastLanguageModel.for_inference(model)

    accuracy = 0
    for example in tqdm(dataset):
        input_ids = tokenizer(example["text"], return_tensors="pt").to("cuda") # WRITE ME
        output_ids = model.generate(**input_ids, max_new_tokens=8, use_cache=True)
        output = tokenizer.decode(output_ids[0]) # WRITE ME

        eos_loc = output.find(tokenizer.eos_token)
        # EOSを出力していなければ間違いとする
        if eos_loc == -1:
            continue

        prediction = output[eos_loc - 1]
        # EOSの直前が数字でなければ必ず間違い
        if not prediction.isdigit():
            continue

        accuracy += int(prediction) == example["answer"]

    return accuracy / len(dataset)

##SFTを行っていないモデルの評価##
torch.cuda.empty_cache()

base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,
)

test_acc = evaluate(test_dataset, base_model, tokenizer)
print(f"Base model test accuracy: {test_acc:.2f}")

##SFTを行ったモデルの評価##
torch.cuda.empty_cache()

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(WORK_DIR / "lora_model"), # 保存したディレクトリを指定
    max_seq_length=MAX_SEQ_LENGTH,
)

test_acc = evaluate(test_dataset, model, tokenizer)
print(f"SFT model test accuracy: {test_acc:.2f}")