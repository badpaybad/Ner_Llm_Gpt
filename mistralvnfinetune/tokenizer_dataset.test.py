import tokenizer_dataset 

from datasets import load_dataset

data=tokenizer_dataset.custom_load_data()

print(data)

from datasets import Dataset

sharegpt_dataset = load_dataset(
    'bkai-foundation-models/vi-self-chat-sharegpt-format')
#[ { "from": "user", "value": "Bạn có thể giúp tôi tìm hiểu về các công nghệ mới trong lĩnh vực máy tính được không?" }, { "from": "gpt", "value": "Tất nhiên, tôi sẽ cố gắng giải đáp những thắc mắc của bạn. Có rất nhiều công nghệ mới trong lĩnh vực máy tính hiện đang được phát triển và áp dụng trên nhiều lĩnh vực khác nhau. Ví dụ như trí tuệ nhân tạo, thực tế ảo, blockchain và IoT." }, { "from": "user", "value": "Vậy trí tuệ nhân tạo là gì?" }, { "from": "gpt", "value": "Trí tuệ nhân tạo là khả năng của máy tính hoặc chương trình máy tính trong thực hiện các tác vụ thông minh như giao tiếp, phân tích và học tập. Nó có thể được sử dụng để tạo ra các hệ thống tự động, trợ giúp tự động và quản lý thông tin." }, { "from": "user", "value": "Thật tuyệt vời! Vậy trí tuệ nhân tạo đã được ứng dụng trong lĩnh vực nào?" }, { "from": "gpt", "value": "Trí tuệ nhân tạo đã được sử dụng rộng rãi trong các lĩnh vực như xe tự lái, dịch thuật tự động, nhận dạng giọng nói và hệ thống hỗ trợ quyết định trong y tế và tài chính." }, { "from": "user", "value": "Wow, thật tuyệt vời! Còn thực tế ảo thì sao?" }, { "from": "gpt", "value": "Thực tế ảo là công nghệ cho phép người dùng tương tác với môi trường máy tính tạo ra một cách thật mà không cần có vật lý. Nó được áp dụng trong trò chơi, giáo dục, kiến trúc và y tế." }, { "from": "user", "value": "Thật thú vị! Tôi nghe nói về blockchain cũng khá nhiều, có thể nói cho tôi hiểu nó được không?" }, { "from": "gpt", "value": "Đương nhiên, blockchain là một công nghệ lưu trữ và truyền tải thông tin một cách an toàn và minh bạch. Nó đang được áp dụng trong các ứng dụng tài chính, quản lý chuỗi cung ứng và đăng ký đất đai." }, { "from": "user", "value": "Vậy nó khác gì với cách thông thường chúng ta lưu trữ thông tin?" }, { "from": "gpt", "value": "Công nghệ blockchain có tính chất phi tập trung, đảm bảo tính toàn vẹn của thông tin và ngăn ngừa việc chỉnh sửa dữ liệu. Điều này giúp nó trở thành một công nghệ tiềm năng trong việc cải thiện tính bảo mật và bảo vệ quyền riêng tư của người dùng." }, { "from": "user", "value": "Cảm ơn trợ lý AI đã giải đáp những thắc mắc của tôi về các công nghệ mới trong lĩnh vực máy tính. Tôi cảm thấy mình đã học được nhiều điều mới và rất thú vị." }, { "from": "gpt", "value": "Tôi vui vì đã giúp được bạn. Còn có gì tôi có thể giúp bạn thêm không?" } ]
train_data = sharegpt_dataset['train'].shuffle(seed=42)\
    .select(range(800))\
    .map(lambda x, i: tokenizer_dataset.tokenize_chat(x, i), remove_columns=["conversations"], with_indices=True)


print(train_data)