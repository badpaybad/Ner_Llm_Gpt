# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="/work/aurora-m-biden-harris-redteamed")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/work/aurora-m-biden-harris-redteamed")
model = AutoModelForCausalLM.from_pretrained("/work/aurora-m-biden-harris-redteamed")


model.eval()

devicetype="cpu"
def response_generate(input_prompt):
    input_ids = tokenizer(input_prompt, return_tensors="pt")
    outputs = model.generate(
        inputs=input_ids["input_ids"].to(devicetype),
        attention_mask=input_ids["attention_mask"].to(devicetype),
        do_sample=True,
        temperature=0.7,
        top_k=50, 
        top_p=0.9,
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    #response = response.split("ASSISTANT:")[-1].strip()
    return response
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

print(response_generate(f"{msg}"))