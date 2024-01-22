
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import time
import numpy as np
from torch.nn import functional as F
import os
from threading import Thread

modelpath="/work/stableai_zephyr16b"
#"stabilityai/stablelm-2-zephyr-1_6b"

print(f"Starting to load the model to memory")
m = AutoModelForCausalLM.from_pretrained(
    modelpath, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
# using CUDA for an optimal experience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
m = m.to(device)
print(f"Sucessfully loaded the model to the memory")


start_message = ""

def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def chat(message, history):
    chat = []
    for item in history:
        chat.append({"role": "user", "content": item[0]})
        if item[1] is not None:
            chat.append({"role": "assistant", "content": item[1]})
    chat.append({"role": "user", "content": message})
    messages = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # Tokenize the messages string
    model_inputs = tok([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(
        tok, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=1000,
        temperature=0.75,
        num_beams=1,
    )
    t = Thread(target=m.generate, kwargs=generate_kwargs)
    t.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        # print(new_text)
        partial_text += new_text
        # Yield an empty string to cleanup the message textbox and the updated conversation history
        #yield partial_text

    return partial_text

# demo = gr.ChatInterface(fn=chat, examples=["hello", "hola", "merhaba"], title="Stable LM 2 Zephyr 1.6b")
# demo.launch()


msgtest="""
you are an expertise in NLP. I give you a scipts conversation between agent that selling insurrance package and client,  They need to conffirm that they understand eache other. you need to parse the infomation inside the conversation i want in json format and in vietnamese totally:  

{"Tên Tư vấn viên":,
"Mã số Đại lý":,
"Sản phẩm bảo hiểm":
"Mức phí đóng hàng năm":,
"Thời hạn đóng phí":,
"Thời hạn hợp đồng":,
"Tên khách hàng":,
"Số điện thoại":,
"địa chỉ khách hàng:"}
Here is the scripts:

	AGENT: Tôi tên Nguyễn Văn A, Mã số đại lý 60000012
	AGENT: Hôm nay tôi thực hiện ghi âm nội dung tư vấn theo yêu cầu của Luật kinh doanh bảo hiểm. Mọi thông tin của cuộc ghi âm sẽ được bảo mật tuân thủ theo Luật bảo mật thông tin  
	AGENT: Hôm nay tôi có tư vấn cho chị Nguyen Thi B về sản phẩm bảo hiểm liên kết đơn vị PRU đầu tư linh hoạt, với mức phí đóng hàng năm là 14,353,353 ngàn
	AGENT: Thời hạn đóng phí 15 năm, thời hạn hợp đồng 20 năm
	AGENT: Với sản phẩm tham gia này, khách hàng sẽ được các quyền lợi như sau:
	AGENT:  Quyền thay đổi lựa chọn quyền lợi bảo hiểm
	AGENT: 	Quyền lợi thưởng, duy trì hợp đồng
	AGENT: Quyền lợi đáo hạn
	AGENT:  	Quyền lợi điều chỉnh hợp đồng trong 21 ngày cân nhắc
	AGENT:  Quyền lợi thay đổi, chọn quỹ đầu tư trong danh sách quỹ liên kết của công ty
AGENT: 	Giá trị quỹ của hợp đồng không được đảm bảo và có thể nhỏ hơn hoặc bằng 0 do phí bảo hiểm không đủ để khấu trừ bảo hiểm rủi rỏ và phí quản lý hợp đồng hoặc do tình hình đầu tư của quỹ. Trong vòng 5 năm hợp đồng đầu tiên, hợp đồng sẽ được duy trì hiệu lực với điều kiện bên mua bảo hiểm đóng đầy đủ phí và đúng hạn phí bản hiểm cơ bản của 5 năm hợp đồng đầu tiên và không thực hiện quyền rút tiền từ Giá trị tài khoản cơ bản
	AGENT:  Bên cạnh đó, tôi xin được lưu ý cho khách hàng răng mọi thông tin khách hàng cung cấp không chính xác sẽ có thể ảnh hưởng đến quyền lợi của khách hàng trong tương lai.
	Customer: 	Tôi tên Nguyễn Thị B, ở địa chỉ 216/9 Hoàng Hoa Thám, Thị Trấn Long Hải, Thành Phố Vũng Tàu, số điện thoại của tôi là 091231020
	Customer: Tôi xác nhận đã được tư vấn viên Nguyễn Văn A tư vấn đầy đủ, và giải thích các điều khoản của hợp đồng. Tôi xác nhận sản phẩm bảo hiểm được tư vấn là phù hợp với nhu cầu tài chính của tôi
"""
import datetime
history=[]

t1=datetime.datetime.now().timestamp()
print(datetime.datetime.now())
print (chat(msgtest,history))
print(datetime.datetime.now().timestamp()-t1)
print(datetime.datetime.now())
print("--------------------------------------------")
t1=datetime.datetime.now().timestamp()
print(datetime.datetime.now())
print (chat(msgtest,history))
print(datetime.datetime.now().timestamp()-t1)
print(datetime.datetime.now())