import transformers
import torch,datetime

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "/work/Meta-Llama-3-8B"
device_type="cpu"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, # change to torch.float16 if you're using V100
    device_map=device_type,
    use_cache=True,
)

system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."

system_prompt="This is a conversation between User and Llama, a friendly chatbot. Llama is helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision."

conversation = [{"role": "system", "content": system_prompt }]

t1= datetime.datetime.now().timestamp()
conversation.append({"role": "user", "content": "How are you today?" })

input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
out_ids = model.generate(
    input_ids=input_ids,
    #max_new_tokens=768,
    max_new_tokens=512,
    pad_token_id=2,
    do_sample=True,
    top_p=0.95,
    top_k=40,
    temperature=0.1,
    repetition_penalty=1.05,
)
assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()    
#conversation.append({"role": "assistant", "content": assistant })

t2= datetime.datetime.now().timestamp()
print(f"elapsed: {t2-t1}")
print(assistant)

def testfromhf():
    model_id = "/work/Meta-Llama-3-8B"

    pipeline = transformers.pipeline(
        "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16,},
        max_new_tokens=256, device_map="cpu"
    )

    res=pipeline("Hey how are you doing today?")
    print(res)