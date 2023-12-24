import py_vncorenlp, os, sys

def getCurrentUserName():
    uname= os.environ.get('USERNAME')
    if uname==None or uname=="":
        uname= os.environ.get('USER')
        
    return uname

workingDir = f"/home/{getCurrentUserName()}/newrobot/ROBOT/src/facerecognition/src"

if getCurrentUserName()=="dunp":
    workingDir = "/work/nuc-newrobot/ROBOT/src/facerecognition/src"

print("api mldlai working dir: " + workingDir)

folderWeight = f"{workingDir}/weights"
os.makedirs(folderWeight, exist_ok=True)

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  ,BitsAndBytesConfig

#model_path = f"{folderWeight}/vinai/PhoGPT-7B5-Instruct"  
model_path="/work/PhoGPT-7B5-Instruct"
os.makedirs(model_path, exist_ok=True)

# ## gpu run with less RAM  # https://huggingface.co/blog/4bit-transformers-bitsandbytes
# config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
# #    bnb_4bit_use_double_quant=True,
# #    bnb_4bit_compute_dtype=torch.bfloat16
#   bnb_4bit_compute_dtype=torch.float16
# )
# model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=config)
# config.init_device = "cuda"


# ## config.attn_config['attn_impl'] = 'triton' # Enable if "triton" installed!  
# ## model = AutoModelForCausalLM.from_pretrained(  
# ##     model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True  
# ## )
# # If your GPU does not support bfloat16:
# model = AutoModelForCausalLM.from_pretrained(model_path, config=config,                                             
#     load_in_4bit=False,
#                                              torch_dtype=torch.float16, trust_remote_code=True)

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True,learned_pos_emb=False)  
config.init_device = "cpu"
model = AutoModelForCausalLM.from_pretrained(model_path, config=config,torch_dtype=torch.float16, trust_remote_code=True)

model.eval()  
  
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  

#tokenizer.bos_token_id = 1
  
PROMPT_TEMPLATE = "### Câu hỏi:\n{instruction}\n\n### Trả lời:"  

# Some instruction examples
# instruction = "Viết bài văn nghị luận xã hội về {topic}"
# instruction = "Viết bản mô tả công việc cho vị trí {job_title}"
# instruction = "Sửa lỗi chính tả:\n{sentence_or_paragraph}"
# instruction = "Dựa vào văn bản sau đây:\n{text}\nHãy trả lời câu hỏi: {question}"
# instruction = "Tóm tắt văn bản:\n{text}"


instruction = "Hoàng Sa, Trường Sa là của nước nào?"
# instruction = "Sửa lỗi chính tả:\nTriệt phá băng nhóm kướp ô tô, sử dụng \"vũ khí nóng\""

input_prompt = PROMPT_TEMPLATE.format_map(  
    {"instruction": instruction}  
)  
  
input_ids = tokenizer(input_prompt, return_tensors="pt")  
  
outputs = model.generate(  
    inputs=input_ids["input_ids"].to(config.init_device),  
    attention_mask=input_ids["attention_mask"].to(config.init_device),  
    do_sample=True,  
    temperature=1.0,  
    top_k=50,  
    top_p=0.9,  
    max_new_tokens=1024,  
    eos_token_id=tokenizer.eos_token_id,  
    pad_token_id=tokenizer.pad_token_id  
)  
  
response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]  
response = response.split("### Trả lời:")[1]


# import py_vncorenlp, os, sys

# curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
# sudo apt-get install git-lfs
# git clone https://oauth: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@huggingface.co/vinai/PhoGPT-7B5-Instruct/

# #https://phoenixnap.com/kb/install-java-ubuntu
# #JAVA_HOME=$(readlink -f /usr/bin/javac | sed "s:/bin/javac::")
# #export PATH=$JAVA_HOME/bin:$PATH

# def getCurrentUserName():
#     uname= os.environ.get('USERNAME')
#     if uname==None or uname=="":
#         uname= os.environ.get('USER')
        
#     return uname

# workingDir = f"/home/{getCurrentUserName()}/newrobot/ROBOT/src/facerecognition/src"

# if getCurrentUserName()=="dunp":
#     workingDir = "/work/nuc-newrobot/ROBOT/src/facerecognition/src"

# print("api mldlai working dir: " + workingDir)

# folderWeight = f"{workingDir}/weights/vncorenlp"
# os.makedirs(folderWeight, exist_ok=True)

# # Automatically download VnCoreNLP components from the original repository
# # and save them in some local machine folder
# py_vncorenlp.download_model(save_dir=folderWeight)

# # Load the word and sentence segmentation component
# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=folderWeight)

# text = "Ông Nguyễn Khắc Chúc  đang làm việc tại Đại học Quốc gia Hà Nội. Bà Lan, vợ ông Chúc, cũng làm việc tại đây."

# output = rdrsegmenter.word_segment(text)

# #print(output)
# # ['Ông Nguyễn_Khắc_Chúc đang làm_việc tại Đại_học Quốc_gia Hà_Nội .', 'Bà Lan , vợ ông Chúc , cũng làm_việc tại đây .']

# import torch
# from transformers import AutoModel, AutoTokenizer

# phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

# # INPUT TEXT MUST BE ALREADY WORD-SEGMENTED!
# #sentence = 'Chúng_tôi là những nghiên_cứu_viên .'  

# for sentence in output:

#     input_ids = torch.tensor([tokenizer.encode(sentence)])

#     with torch.no_grad():
#         features = phobert(input_ids)  # Models outputs are now tuples
        
#         print(features)
        
#         for tok in features:
#             print(str(tok))


