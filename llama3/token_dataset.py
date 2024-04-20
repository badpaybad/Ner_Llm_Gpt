import gc
import os

import torch
# import wandb
# pip install wandb google.colab
# pip install -U git+https://github.com/huggingface/trl
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)


from trl import ORPOConfig, ORPOTrainer, setup_chat_format

# pip install  flash-attn
# pip install -U transformers datasets accelerate peft trl bitsandbytes wandb --progress-bar off
# Model
base_model = "/work/Meta-Llama-3-8B"
new_model = "OrpoLlama-3-8B"


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)


dataset_name = "mlabonne/orpo-dpo-mix-40k"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=42).select(
    range(1))  # Only use 1000 samples for quick demo

print(dataset)


def format_chat_template(row):
    print("--------------------------------")
    print(row["chosen"])
    print("--------------------------------")
    row["chosen"] = tokenizer.apply_chat_template(
        row["chosen"], tokenize=False)
    print(row["chosen"])
    print("--------------------------------")
    row["rejected"] = tokenizer.apply_chat_template(
        row["rejected"], tokenize=False)
    return row


dataset = dataset.map(
    format_chat_template,
    num_proc=os.cpu_count(),
)
dataset = dataset.train_test_split(test_size=0.01)

print(dataset)
