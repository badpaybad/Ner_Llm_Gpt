import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."

tokenizer = AutoTokenizer.from_pretrained('/work/Mistral-7B-v0.1')
model = AutoModelForCausalLM.from_pretrained(
    '/work/Mistral-7B-v0.1',
    torch_dtype=torch.bfloat16, # change to torch.float16 if you're using V100
    device_map="auto",
    use_cache=True,
)

conversation = [{"role": "system", "content": system_prompt }]
while True:
    human = input("Human: ")
    if human.lower() == "reset":
        conversation = [{"role": "system", "content": system_prompt }]
        print("The chat history has been cleared!")
        continue

    conversation.append({"role": "user", "content": human })
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
    
    out_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=768,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        temperature=0.1,
        repetition_penalty=1.05,
    )
    assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
    print("Assistant: ", assistant) 
    conversation.append({"role": "assistant", "content": assistant })
