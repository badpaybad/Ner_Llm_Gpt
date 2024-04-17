
from datasets import Dataset
from datasets import load_dataset
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model


tokenizer_path = '/work/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat'
model_path= '/work/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat'

conversation_template = {
    "conversations": [
        [
            {
                "role": "system",
                "content": ""
            },
            {
                "role": "user",
                "content": ""
            },
            {
                "role": "assistant",
                "content": ""
            }
        ]
    ]
}


def generate_system_prompt(i):
    system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn."
    if i % 2 == 0:
        system_prompt += "\nCâu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
    if i % 5 == 0:
        system_prompt += "\nNếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."
    return system_prompt


def tokenizer():
    return AutoTokenizer.from_pretrained(tokenizer_path)


def tokenize_chat(input, i,tokenizer=tokenizer()):
    """_summary_

    Args:
        input (openai chatml): [{{role:system , conent:},{role:user , conent:},{role:assistant , conent:}},...]
        i (int): random generate system prompt

    Returns:
        AutoTokenizer: AutoTokenizer
    """
    #input format: [ { "from": "user", "value": "Bạn có thể giúp tôi tìm hiểu về các công nghệ mới trong lĩnh vực máy tính được không?" }, { "from": "gpt", "value": "Tất nhiên, tôi sẽ cố gắng giải đáp những thắc mắc của bạn. Có rất nhiều công nghệ mới trong lĩnh vực máy tính hiện đang được phát triển và áp dụng trên nhiều lĩnh vực khác nhau. Ví dụ như trí tuệ nhân tạo, thực tế ảo, blockchain và IoT." }, { "from": "user", "value": "Vậy trí tuệ nhân tạo là gì?" }, { "from": "gpt", "value": "Trí tuệ nhân tạo là khả năng của máy tính hoặc chương trình máy tính trong thực hiện các tác vụ thông minh như giao tiếp, phân tích và học tập. Nó có thể được sử dụng để tạo ra các hệ thống tự động, trợ giúp tự động và quản lý thông tin." }, { "from": "user", "value": "Thật tuyệt vời! Vậy trí tuệ nhân tạo đã được ứng dụng trong lĩnh vực nào?" }, { "from": "gpt", "value": "Trí tuệ nhân tạo đã được sử dụng rộng rãi trong các lĩnh vực như xe tự lái, dịch thuật tự động, nhận dạng giọng nói và hệ thống hỗ trợ quyết định trong y tế và tài chính." }, { "from": "user", "value": "Wow, thật tuyệt vời! Còn thực tế ảo thì sao?" }, { "from": "gpt", "value": "Thực tế ảo là công nghệ cho phép người dùng tương tác với môi trường máy tính tạo ra một cách thật mà không cần có vật lý. Nó được áp dụng trong trò chơi, giáo dục, kiến trúc và y tế." }, { "from": "user", "value": "Thật thú vị! Tôi nghe nói về blockchain cũng khá nhiều, có thể nói cho tôi hiểu nó được không?" }, { "from": "gpt", "value": "Đương nhiên, blockchain là một công nghệ lưu trữ và truyền tải thông tin một cách an toàn và minh bạch. Nó đang được áp dụng trong các ứng dụng tài chính, quản lý chuỗi cung ứng và đăng ký đất đai." }, { "from": "user", "value": "Vậy nó khác gì với cách thông thường chúng ta lưu trữ thông tin?" }, { "from": "gpt", "value": "Công nghệ blockchain có tính chất phi tập trung, đảm bảo tính toàn vẹn của thông tin và ngăn ngừa việc chỉnh sửa dữ liệu. Điều này giúp nó trở thành một công nghệ tiềm năng trong việc cải thiện tính bảo mật và bảo vệ quyền riêng tư của người dùng." }, { "from": "user", "value": "Cảm ơn trợ lý AI đã giải đáp những thắc mắc của tôi về các công nghệ mới trong lĩnh vực máy tính. Tôi cảm thấy mình đã học được nhiều điều mới và rất thú vị." }, { "from": "gpt", "value": "Tôi vui vì đã giúp được bạn. Còn có gì tôi có thể giúp bạn thêm không?" } ]
    #print(generate_system_prompt(i))
    # convert to chatml, openai standard : conversation_template
    conversation = [{'role': 'system', 'content': generate_system_prompt(i)}]
    for msg in input['conversations']:
        output = {'role': 'user', 'content': msg['value']}
        if msg['from'] == 'gpt':
            output['role'] = 'assistant'
        conversation.append(output)
    formatted = tokenizer.apply_chat_template(conversation, tokenize=False)
    return tokenizer(formatted)


def custom_load_data(fullpathfile="nuiblog.data.json",tokenizer=tokenizer()):

    f = open(fullpathfile)
    data = json.load(f)
    f.close()
    print('Number of conversations', len(data))
    train_data = []
    for i, conversation in enumerate(data):
        text = tokenizer.apply_chat_template(conversation, tokenize=False)
        train_data.append({'text': text})
    train_data = Dataset.from_list(train_data).map(
        lambda x: tokenizer(x['text']))

    return train_data
