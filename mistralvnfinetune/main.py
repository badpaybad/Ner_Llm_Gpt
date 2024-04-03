import os,torch

os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"

from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging

from transformers import TrainingArguments, Trainer, TensorBoardCallback
"""
pip install -U bitsandbytes
pip install -U transformers
pip install -U peft
pip install -U accelerate
pip install -U trl
pip install -U datasets
pip install tensorboard
pip install tensorboardX

"""

base_model = "/work/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat"
dataset_name = "mlabonne/guanaco-llama2-1k"
new_model = "mistral_7b_guanaco"

#Importing the dataset
dataset = load_dataset(dataset_name, split="train")
dataset["text"][100]

device_type="cpu"

# wandb.login(key = secret_wandb)
# run = wandb.init(
#     project='Fine tuning mistral 7B', 
#     job_type="training", 
#     anonymous="allow"
# )
if device_type=="cpu":
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            # load_in_4bit=True,
            # quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device_type,
            trust_remote_code=True,
    )
else:
    bnb_config = BitsAndBytesConfig(  
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )
    model = AutoModelForCausalLM.from_pretrained(
            base_model,
            #load_in_4bit=True,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map=device_type,
            trust_remote_code=True,
    )
    
model.config.use_cache = False # silence the warnings
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token

from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
)
model = get_peft_model(model, peft_config)


training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    #report_to="wandb"
    logging_dir='./logs', 
    logging_strategy='steps',      
    device=device_type
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= None,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= False,
    callbacks=[TensorBoardCallback()], 
)

trainer.train()

trainer.model.save_pretrained(new_model)
#wandb.finish()
model.config.use_cache = True

#tensorboard --logdir ./logs
