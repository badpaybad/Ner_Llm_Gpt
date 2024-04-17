from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from accelerate import Accelerator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os
import torch
# import wandb
import platform
import warnings
from datasets import load_dataset
from trl import SFTTrainer

import mistralvnfinetune.tokenizer_dataset as tokenizer_dataset

hf_token = ''
wnb_token = ''
wnb_name = 'vistral-chatml'
resume_from_checkpoint = False
output_dir = 'vistral-chatml'
tokenizer_dataset.model_path = '/work/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat'
tokenizer_dataset.tokenizer_path = '/work/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat'

#######################################################
# DATASET

tokenizer = tokenizer_dataset.tokenizer()

sharegpt_dataset = load_dataset(
    'bkai-foundation-models/vi-self-chat-sharegpt-format')
#[ { "from": "user", "value": "Bạn có thể giúp tôi tìm hiểu về các công nghệ mới trong lĩnh vực máy tính được không?" }, { "from": "gpt", "value": "Tất nhiên, tôi sẽ cố gắng giải đáp những thắc mắc của bạn. Có rất nhiều công nghệ mới trong lĩnh vực máy tính hiện đang được phát triển và áp dụng trên nhiều lĩnh vực khác nhau. Ví dụ như trí tuệ nhân tạo, thực tế ảo, blockchain và IoT." }, { "from": "user", "value": "Vậy trí tuệ nhân tạo là gì?" }, { "from": "gpt", "value": "Trí tuệ nhân tạo là khả năng của máy tính hoặc chương trình máy tính trong thực hiện các tác vụ thông minh như giao tiếp, phân tích và học tập. Nó có thể được sử dụng để tạo ra các hệ thống tự động, trợ giúp tự động và quản lý thông tin." }, { "from": "user", "value": "Thật tuyệt vời! Vậy trí tuệ nhân tạo đã được ứng dụng trong lĩnh vực nào?" }, { "from": "gpt", "value": "Trí tuệ nhân tạo đã được sử dụng rộng rãi trong các lĩnh vực như xe tự lái, dịch thuật tự động, nhận dạng giọng nói và hệ thống hỗ trợ quyết định trong y tế và tài chính." }, { "from": "user", "value": "Wow, thật tuyệt vời! Còn thực tế ảo thì sao?" }, { "from": "gpt", "value": "Thực tế ảo là công nghệ cho phép người dùng tương tác với môi trường máy tính tạo ra một cách thật mà không cần có vật lý. Nó được áp dụng trong trò chơi, giáo dục, kiến trúc và y tế." }, { "from": "user", "value": "Thật thú vị! Tôi nghe nói về blockchain cũng khá nhiều, có thể nói cho tôi hiểu nó được không?" }, { "from": "gpt", "value": "Đương nhiên, blockchain là một công nghệ lưu trữ và truyền tải thông tin một cách an toàn và minh bạch. Nó đang được áp dụng trong các ứng dụng tài chính, quản lý chuỗi cung ứng và đăng ký đất đai." }, { "from": "user", "value": "Vậy nó khác gì với cách thông thường chúng ta lưu trữ thông tin?" }, { "from": "gpt", "value": "Công nghệ blockchain có tính chất phi tập trung, đảm bảo tính toàn vẹn của thông tin và ngăn ngừa việc chỉnh sửa dữ liệu. Điều này giúp nó trở thành một công nghệ tiềm năng trong việc cải thiện tính bảo mật và bảo vệ quyền riêng tư của người dùng." }, { "from": "user", "value": "Cảm ơn trợ lý AI đã giải đáp những thắc mắc của tôi về các công nghệ mới trong lĩnh vực máy tính. Tôi cảm thấy mình đã học được nhiều điều mới và rất thú vị." }, { "from": "gpt", "value": "Tôi vui vì đã giúp được bạn. Còn có gì tôi có thể giúp bạn thêm không?" } ]
train_data = sharegpt_dataset['train'].shuffle(seed=42)\
    .select(range(800))\
    .map(lambda x, i: tokenizer_dataset.tokenize_chat(x, i), remove_columns=["conversations"], with_indices=True)

# train_data= token_dataset.custom_load_data()

#######################################################
# SETUP

# wandb.login(key=wnb_token)
# wandb.init(name=wnb_name)
# use custom tokenizer instead of one comes from the model
# tokenizer = AutoTokenizer.from_pretrained(
#  MODEL,
#  add_eos_token=False,
#  add_bos_token=False,
#  token=hf_token,
# )
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    tokenizer_dataset.model_path,
    device_map="auto",
    token=hf_token,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

#######################################################
# LORA CONFIG
"""
#https://blog.ngxson.com/fine-tune-model-viet-truyen-phong-cach-nam-cao/
Lưu ý khi chọn lora rank, alpha và learning rate:

    rank 8 và alpha 16: Nếu bạn muốn fine tune nhẹ nhàng, ví dụ chỉ thay đổi văn phong của model thôi chẳng hạn. Ngoài ra, rank thấp thì train càng nhanh.
    rank 16 và alpha 32: Nếu bạn bắt đầu muốn “ghi đè” kiến thức đã có, ví dụ nếu model lúc nào cũng bảo “Là một AI, tôi không thể blah blah”, thì bạn có thể ghi đè kiến thức “Tôi là AI” bằng cái khác, ví dụ ở https://huggingface.co/ngxson/vistral-meow mình ghi đè thành “Tôi là Meow”.
    Warmup steps: Mình để chỉ 2 hay 5 steps thôi, nhưng thường người ta để khá cao, vài chục thậm chí vài trăm. Cá nhân mình thấy là: Nếu loss mãi không xuống thì tăng warmup step lên và giảm learning rate xuống.
    Learning rate: Hiện tại mình để 2.5e-5, nhưng nếu loss mãi không xuống thì bạn có thể chia 2, hay chia 10 số đó.

Với ví du này (viết văn Nam Cao), mình lấy rank 16, alpha 32, warmup 2 và learning rate 2.5e-5
"""

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

accelerator = Accelerator()
model = accelerator.prepare_model(model)


#######################################################
# TRAIN

trainer = Trainer(
    model=model,
    train_dataset=train_data,
    args=TrainingArguments(
        #report_to='wandb',
        warmup_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_train_epochs=4,
        learning_rate=2.5e-5,
        logging_steps=1,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_steps=10,
        save_total_limit=4,
        output_dir=output_dir
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False

trainer.train(resume_from_checkpoint=resume_from_checkpoint)
