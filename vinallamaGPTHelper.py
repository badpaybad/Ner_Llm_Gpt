import argparse
# Create the argument parser
parser = argparse.ArgumentParser()

## Add arguments
# parser.add_argument('-n', '--model_name', type=str, help='Specify a model name', default='vilm/vietcuna-3b', choices=['vilm/vietcuna-3b', 'vilm/vietcuna-7b'])
# parser.add_argument('--four-bit', action='store_true', help='Whether to use 4bit')

# args = parser.parse_args()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer, BitsAndBytesConfig

# if args.model_name == 'vietcuna-3b':
#     model_name = 'vilm/vietcuna-3b'
# elif args.model_name == 'vietcuna-7b':
#     model_name = 'vilm/vietcuna-7b-alpha'
# else:
#     raise ValueError("Unsupported model_name. Please choose either 'vietcuna-3b' or 'vietcuna-7b'.")

model_name='/work/vinallama-7b'
model_name='/work/vinallama-2.7b'

print(f"Starting to load the model {model_name} into memory")

# https://huggingface.co/blog/4bit-transformers-bitsandbytes
# nf4_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16
# )

# model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)

m = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=False,
    torch_dtype=torch.bfloat16,
    #device_map={"": 0} # will take 5 minue to get response 
    device_map="cpu"  #will take about 2hour to get response
    #device_map="auto", 
    #offload_folder="offload",
)

tok = AutoTokenizer.from_pretrained(model_name)
tok.bos_token_id = 1

stop_token_ids = [2]

print(f"Successfully loaded the model {model_name} into memory")

import datetime
import os
from threading import Event, Thread
from uuid import uuid4

import gradio as gr
import requests

max_new_tokens = 512 if model_name == 'vilm/vietcuna-3b' else 1024

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

stop = StopOnTokens()
temperature=0.7
top_p=0.9
top_k=0
repetition_penalty = 1.0
import datetime
def generateText( messages):   
    
    print(f"begin {datetime.datetime.now()}")
    #messages="Hoàng Sa, Trường Sa là của nước nào?"
    print("begin generate text")
    # Tokenize the messages string
    input_ids = tok(messages, return_tensors="pt").input_ids
    input_ids = input_ids.to(m.device)
    streamer = TextIteratorStreamer(tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature ,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([stop]),
    )

    m.generate(**generate_kwargs)

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text

    print(f"done {datetime.datetime.now()}")
    print(partial_text)
    return partial_text

msg="""
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

generateText(msg)

#generateText("Hoàng Sa, Trường Sa là của nước nào?")

"""
Đây là câu hỏi được đặt ra trong một buổi thảo luận về chủ quyền lãnh thổ Việt Nam trên Biển Đông do Báo VietNamNet tổ chức chiều qua tại Hà Nội.
Các chuyên gia và nhà nghiên cứu cho rằng, Hoàng Sa và Trường Sa là của Việt Nam. Tuy nhiên, một số ý kiến cho rằng, đây chỉ là khu vực tranh chấp, không thể xem là của nước nào.
Chuyên gia khoa học quân sự Nguyễn Minh Thuyết (Học viện Chính trị Quân sự Quốc gia Hồ Chí Minh) cho rằng, Hoàng Sa, Trường Sa là của Việt Nam.
“Các đảo này có lịch sử, có tính pháp lý, có hệ thống văn bản pháp lý. Các đảo này nằm trong vùng đặc quyền kinh tế, thềm lục địa của Việt Nam. Không có chuyện tranh chấp, không có chuyện tranh chấp lãnh thổ, không có chuyện tranh chấp lãnh thổ với Trung Quốc. Chúng ta khẳng định chủ quyền, không có chuyện tranh chấp lãnh thổ với Trung Quốc”, ông Thuyết khẳng định.
Đồng quan điểm, ông Nguyễn Minh Thuyết cho rằng, các đảo này có lịch sử, có tính pháp lý, có hệ thống văn bản pháp lý. Các đảo này nằm trong vùng đặc quyền kinh tế, thềm lục địa của Việt Nam.
“Không có chuyện tranh chấp, không có chuyện tranh chấp lãnh thổ, không có chuyện tranh chấp lãnh thổ với Trung Quốc. Chúng ta khẳng định chủ quyền, không có chuyện tranh chấp lãnh thổ với Trung Quốc”, ông Thuyết khẳng định.
Phó giáo sư, Tiến sĩ Trần Đình Thiên, Viện trưởng Viện Kinh tế Việt Nam, khẳng định, Hoàng Sa và Trường Sa là của Việt Nam.
“Tất cả các đảo này đều có lịch sử, có tính pháp lý, có hệ thống văn bản pháp lý. Các đảo này nằm trong vùng đặc quyền kinh tế, thềm lục địa của Việt Nam. Chúng ta khẳng định chủ quyền, không có chuyện tranh chấp, không có chuyện tranh chấp lãnh thổ, không có chuyện tranh chấp lãnh thổ với Trung Quốc”, ông Thiên nói.
“Từng có tranh chấp, chúng ta đã giải quyết xong. Mọi tranh chấp đã giải quyết ổn thỏa, không có chuyện tranh chấp lãnh thổ. Tất cả các đảo này đều có lịch sử, có tính pháp lý, có hệ thống văn bản pháp lý. Các đảo này nằm trong vùng đặc quyền kinh tế, thềm lục địa của Việt Nam. Chúng ta khẳng định chủ quyền, không có chuyện tranh chấp lãnh thổ với Trung Quốc”, ông Thiên khẳng định.
Phó giáo sư, Tiến sĩ Trần Đình Thiên, Viện trưởng Viện Kinh tế Việt Nam, khẳng định, Hoàng Sa và Trường Sa là của Việt Nam.
“Chúng ta có đầy đủ cơ sở pháp lý, chúng ta khẳng định chủ quyền. Tất cả các đảo này đều có lịch sử, có tính pháp lý, có hệ thống văn bản pháp lý. Các đảo này nằm trong vùng đặc quyền kinh tế, thềm lục địa của Việt Nam. Chúng ta khẳng định chủ quyền, không có chuyện tranh chấp, không có chuyện tranh chấp lãnh thổ, không có chuyện tranh chấp lãnh thổ với Trung Quốc”, ông Thiên khẳng định.
“Chúng ta có đầy đủ cơ sở pháp lý, chúng ta khẳng định chủ quyền. Tất cả các đảo này đều có lịch sử, có tính pháp lý, có hệ thống văn bản pháp lý. Các đảo này nằm trong vùng đặc quyền kinh tế, thềm lục địa của Việt Nam. Chúng ta khẳng định chủ quyền, không có chuyện tranh chấp, không có chuyện tranh chấp lãnh thổ, không có chuyện tranh chấp lãnh thổ với Trung Quốc”, ông Thiên khẳng định.
Hoàng Sa, Trường Sa là của Việt Nam
Chuyên gia khoa học quân sự Nguyễn Minh Thuyết (Học viện Chính trị Quân sự Quốc gia Hồ Chí Minh) cho rằng, Hoàng Sa, Trường Sa là của Việt Nam.
“Các đảo này có lịch sử, có tính pháp lý, có hệ thống văn bản pháp lý. Các đảo này nằm trong vùng đặc quyền kinh tế, thềm lục địa của Việt Nam. Không có chuyện tranh chấp, không có chuyện tranh chấp lãnh thổ, không có chuyện tranh chấp lãnh thổ với Trung Quốc. Chúng ta khẳng định chủ quyền, không có chuyện tranh chấp lãnh thổ với Trung Quốc”, ông Thuyết khẳng định.
Phó giáo sư, Tiến sĩ Trần Đình Thiên, Viện trưởng Viện Kinh tế Việt Nam, khẳng định, Hoàng Sa và Trường Sa là của Việt Nam.
“Tất cả các đảo này đều có lịch sử, có tính pháp lý, có hệ thống văn bản pháp lý. Các đảo này nằm trong vùng đặc quyền kinh tế, thềm lục địa của Việt Nam. Chúng ta khẳng định chủ quyền, không có chuyện tranh chấp, không có chuyện tranh chấp lãnh thổ, không có chuyện tranh chấp lãnh thổ với Trung Quốc”, ông Thiên nói.
Phó giáo sư, Tiến sĩ Trần Đình Thiên, Viện trưởng Viện Kinh tế Việt Nam, khẳng định
"""