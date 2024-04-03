import os
#os.environ["HIP_LAUNCH_BLOCKING"]="1"
# 
# pip install -U torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
os.environ["PYTORCH_ROCM_ARCH"] = "gfx1103" 
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0" 
os.environ["HIP_VISIBLE_DEVICES"] = "0" 
# os.environ["ROCM_PATH"] = "/opt/rocm" 
# Use a pipeline as a high-level helper
# pip3 install -U numpy torch torchvision torchaudio soundfile sox --extra-index-url https://download.pytorch.org/whl/cpu

from transformers import pipeline

modelFolder="/work/TinyLlama-1.1B-Chat-v1.0"
import torch
from transformers import pipeline
devicemap="cpu" # auto cpu gpu
pipe = pipeline("text-generation", model=modelFolder, torch_dtype=torch.bfloat16, device_map=devicemap)

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "Bạn là giáo viên thông thái và rất yêu mến học sinh",
    },
    {"role": "user", "content": "Kể một câu chuyện cười về con mèo"},
]
prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
print(outputs[0]["generated_text"])