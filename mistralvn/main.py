from fastapi.security import HTTPBearer, OAuth2PasswordBearer
import uuid
from asyncio import sleep as async_sleep
from contextlib import asynccontextmanager
from anyio import CapacityLimiter
from anyio.lowlevel import RunVar
from fastapi import FastAPI, File, Form, UploadFile, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi import Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
import os
import datetime
import math
import sys
import threading
import re
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
import torch

# pip3 install -U torch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 --index-url https://download.pytorch.org/whl/cpu

# pip3 install install torch==2.1.0+cpu torchvision==0.16.0 torchaudio==2.1.0  --index-url https://download.pytorch.org/whl/cpu
# pip3 install torch-npu==2.1.0
# import torch_npu
# insert at 1, 0 is the script path (or '' in REPL)
____workingDir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(1, ____workingDir)

print("____workingDir", ____workingDir)

os.environ["PYTORCH_HIP_ALLOC_CONF"] = "expandable_segments:True"
# path = os.environ["HIP_LAUNCH_BLOCKING"]="1"
# os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
# os.environ["HIP_VISIBLE_DEVICES"] = "0"
# os.environ["AMD_SERIALIZE_KERNEL"] = "3"
# os.environ["TORCH_USE_HIP_DSA"] = "1"

# os.environ["PYTORCH_ROCM_ARCH"] = "gfx1103"
# os.environ["HCC_AMDGPU_TARGET"] = "gfx1103"
# os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

print(f'CUDA support: {torch.cuda.is_available()} (Should be "True")')
print(f'CUDA version: {torch.version.cuda} (Should be "None")')
print(f"HIP version: {torch.version.hip} (Should contain value)")

print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
# print(torch.cuda.memory_summary())

class EmbeddingRequest(BaseModel):
    input: str
    encoding_format: "float"
    model: str = ""


class Embedding(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int = 0


class Usage(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int = 0


class EmbeddingResponse(BaseModel):
    object: str="List"
    data: List[Embedding]
    model: str
    usage: Usage


class Message(BaseModel):
    role: str="user"
    content: str="Hello"


class ConversationRequest(BaseModel):
    conversationid: str = f"{uuid.uuid4()}"


class ChatRequest(BaseModel):
    model: str = "Vistral-7B-Chat"
    messages: List[Message]
    temperature: float = 1
    max_tokens: int = 1024
    do_sample: bool = True
    top_p: Optional[float] = 0.95
    top_k: Optional[int] = 40
    repetition_penalty: float = 0.95
    conversationid: str = f"{uuid.uuid4()}"
    tools: Optional[str] = None
    requestid: str = f"{uuid.uuid4()}"
    skip_special_tokens: bool = True


class Choice(BaseModel):
    message: Message
    finish_reason: Optional[str] = None
    index: int = 0

class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    conversationid: str = f"{uuid.uuid4()}"
    tools: Optional[str] = None
    requestid: str = f"{uuid.uuid4()}"
    elapsed_in_milis: float = 0


_http_port = str(sys.argv[1])

device_type = "cuda"
if len(sys.argv) > 2:
    device_type = str(sys.argv[2])

if device_type == None or device_type == "":
    device_type = "cpu"

print(f"device_type try to use: {device_type}")

# torch.cuda.empty_cache()
# import gc
# gc.collect()
# torch.cuda.memory_summary(device=None, abbreviated=False)

# import testrocm

# testrocm.test_amd_rocm()

system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."

# pip3 install -U --pre torch torchvision torchaudio transformers --index-url https://download.pytorch.org/whl/nightly/rocm6.0
"""
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.1/ubuntu/jammy/amdgpu-install_6.1.60100-1_all.deb
#https://repo.radeon.com/amdgpu-install/6.0.3/ubuntu/jammy/
sudo apt install ./amdgpu-install_6.1.60100-1_all.deb
sudo dpkg -i amdgpu-install_6.1.60100-1_all.deb
sudo amdgpu-install -y --usecase=hiplibsdk,rocmdev,dkms,mllib,mlsdk 

ln -s /var/lib/dkms/amdgpu/6.3.6-1739731.22.04/source /var/lib/dkms/amdgpu/6.3.6-1718217.22.04/source

"""

# #system_prompt+="You are a supervisor at the customer care switchboard of a Vietnamese bank. Your job is to monitor whether trained staff are following procedures when speaking with customers\n"
# msg="You need to extract the criteria export code and conversation topic code as JSON format with the following structure:{\"Criterias\":[{\"Code\":\"agent-self-introduce\",\"Name\": \"Điện thoại viên chào hỏi khách hàng\", \"Value\": the content that agent introduce herself,\"Times\":array of all the start times in seccond when agent introduce herself },{\"Code\":\"customer-name\",\"Name\": \"Tên khách hàng\", \"Value\": customer name,\"Times\":array of all the start times in seccond when agent use customer name},{\"Code\":\"customer-name\",\"Name\": \"Tên khách hàng\", \"Value\": customer name,\"Times\":array of all the start times in seccond when agent use customer name},{\"Code\":\"customer-name\",\"Name\": \"Tên khách hàng\", \"Value\": customer name,\"Times\":array of all the start times in second when agent use customer name},{\"Code\":\"customer-complain-agent-bad-attitude\",\"Name\": \"Khách hàng phàn nàn thái độ điện thoại viên\", \"Value\": customer complain that agent has bad attitude,\"Times\":array of all the start times in second when customer complain that agent has bad attitude},{\"Code\":\"customer-complain-agent-voice\",\"Name\": \"Khách hàng phàn nàn âm lượng điện thoại viên\", \"Value\": customer complain that agent voice is difficult to hear,\"Times\":aray of all the start times in second when customer complain that agent voice is difficult to hear},{\"Code\":\"password-info\",\"Name\": \"Thông tin mật khẩu\", \"Value\": customer secret password info,\"Times\":aray of all the start times in second when customer secret password info is found}],\"Topic\": conversation topic } In there,\n- Criterias is a list of criteria. Each criterion includes:\n\t+ Code is the criteria code.\n\t+ Name is the criteria name.\n\t+ Text is entities extracted\n\t+ Times array of all the start times in seccond when the saying appears in conversation\n- Topic is the topic of the conversation\n\nThis is 4 topics codes include:\n- Nghiệp vụ thẻ, included keywords: đóng thẻ,mở thẻ,phát hành,câu hỏi bảo mật,hạn mức - Nghiệp vụ vay, included keywords: khoản vay,hạn thanh toán,ngày trả nợ,thu nợ,quét nợ,mở vay,ngày vay,ngày trả nợ - Nghiệp vụ NEO, included keywords: mật khẩu,tên đăng nhập,lỗi chuyển tiền,không đăng nhập - Nghiệp vụ tiết kiệm, included keywords: lãi suất,sổ,số tiền sổ,ngày gửi,tất toán sổ,giải toả sổ,số sổ\n\nYou should only respond in a JSON structure. If any criteria or topics cannot be extracted, return \"\". Get as much information as possible\n\nThis is a conversation that needs to be handle\nNote: Number inside () is start time\nAgent(2.319999933242798): dạ danh em ly xin nghe Agent(3.7899999618530273): em có thể hỗ trợ anh chị thông tin gì ạ Customer(5.800000190734863): ừ em ơi Customer(6.679999828338623): chị có một cái thẻ tín dụng ờ classic nó mobifone ý Customer(9.889999389648438): mà chị đang tìm hiểu về có nhu cầu muốn nâng lên cái mở cái thẻ ly ly được không em Agent(14.769999504089355): ờ chị vui lòng cho em xin lại tên của chị để em tiện trao đổi ạ Customer(18.489999771118164): ờ chị liên liên em ạ Agent(20.169998168945312): vâng Agent(20.8700008392334): chị cho em xin lại chứng minh thư để em kiểm tra thêm thông tin ạ Customer(24.10999870300293): ờ đúng rồi chờ chị một chút nhá Customer(26.159997940063477): chứng minh thư là Agent(26.409997940063477): dạ vâng Customer(27.729999542236328): 123 Customer(28.90999984741211): 456789 em ơi Agent(31.23999786376953): dạ vâng Agent(32.09000015258789): chị vui lòng đợi máy giúp em trong giây lát nhá Customer(34.59000015258789): ok Customer(58.269996643066406): nó bảo phải chờ máy Customer(70.37000274658203): biết được 30 giây Customer(75.91999816894531): alo em ơi được chưa em ơi Agent(78.41000366210938): à dạ cảm ơn chị chờ máy xin lỗi để chị chờ máy hơi lâu ạ Agent(81.41000366210938): ờ đối với trường hợp của Agent(83.22999572753906): chị hiên thì Agent(84.60999298095703): hiện tại em kiểm tra thông tin thì chị nạp chưa nằm trong danh sách được mở sim thẻ Agent(88.18999481201172): đây là rất tiếc bên em chưa thể hỗ trợ mình mở thêm thẻ rồi chị ạ Customer(91.9000015258789): thế hả Agent(93.05999755859375): dạ vâng Customer(94.61000061035156): mình cần trách danh sách hả em Agent(96.41999816894531): dạ đúng rồi nếu trong trường hợp này mình có nhu cầu sử dụng ờ Agent(100): cái thẻ Agent(100.75): setup ý ạ Agent(101.41000366210938): thì chị có thể đóng lại cái thẻ classic mobifone giúp em Agent(104.75): và vui lòng chờ giúp em sau 30 ngày kể từ thời điểm đóng thẻ Agent(107.47999572753906): hợp đồng đóng thẻ của mình sẽ trình hồ sơ để mở thêm thẻ giáp bát chị nhé Customer(112.12000274658203): ừ Customer(113.02999114990234): tức là có nghĩa bây giờ chị phải đóng cái thẻ của chị đang xài Agent(115.56999969482422): dạ đúng rồi Customer(115.77999877929688): sau đó rồi chị 30 ngày sau thì chị mới mở được cái dây thẻ mới Agent(119.86000061035156): dạ đúng rồi ạ Agent(119.96000061035156): chị cho em xin số otp gửi về Customer(120.78999938964844): 76342///Customer(120.88999938964844): chỉ mỗi cách này thôi hả em Agent(122.43000030517578): dạ đúng rồi ạ chỉ có mỗi cách này thôi ạ bởi vì hiện tại em kiểm tra thông tin thì rất tiếc là chị không nằm trong danh sách mở thêm thẻ bên em Customer(129.2899932861328): sắp hai nhiều khách hàng có nhu cầu muốn sử dụng thêm thì phải Customer(132.6199951171875): trong danh sách mới mở được à Agent(134.1999969482422): dạ vâng đúng rồi ạ Agent(136.36000061035156): thì không biết là hiện tại chị có muốn đóng thẻ classic không ạ Customer(136.7100067138672): ừ Customer(139.5800018310547): thôi để chị xem Customer(141.55999755859375): em thêm nhá chị tìm hiểu đã Customer(142.97000122070312): chị nắm được thông tin rồi Agent(144.6999969482422): dạ vâng không biết là chị hiên còn thắc mắc thông tin nào khác nữa không chị Customer(147.80999755859375): ừ thôi chị không có thông tin gì nữa đâu Customer(149.84999084472656): chị cảm ơn Agent(150.58999633789062): dạ vâng thế có vấn đề gì thì yên vui lòng liên hệ lại sau giúp em ạ Agent(153.45999145507812): cảm ơn chị em chào chị hen Customer(154.69000244140625): ok "

# msg="""
# từ đoạn văn sau: "Cô Thu là em gái của mẹ. Năm nay, cô ba mươi tuổi. Cô sống cùng với bà ngoại. Cô là một giáo viên dạy Toán. Cô rất xinh đẹp và dễ mến. Cuối tuần, tôi thường sang chơi với cô. Tôi được nghe cô kể nhiều chuyện hay. Thỉnh thoảng, cô còn hướng dẫn tôi làm bài tập. Cô giống như một người bạn của tôi vậy. Tôi quý mến và kính trọng cô lắm!" trích xuất thông tin dạng mẫu json {
# "name":"tên người ở đây",
# "hometown":"tên quê quán",
# "summary":"tóm tắt đoạn văn"
# }
# """
# modelpath = "/work/shared/vicuna-7b-v1.5"
# modelpath = "/work/shared/gemma-2-2b"
modelpath = "/work/shared/Vistral-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(modelpath)
model = AutoModelForCausalLM.from_pretrained(
    modelpath,
    # torch_dtype=torch.bfloat16,  # change to torch.float16 if you're using V100
    device_map=device_type,
    use_cache=False,
)


# Move the model to the NPU
device = torch.device(device_type)
model.to(device)

conversation = [{"role": "system", "content": system_prompt}]

# @webApp.on_event("startup")
# def startup():
#     print("start")
#     RunVar("_default_thread_limiter").set(CapacityLimiter(2))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    RunVar("_default_thread_limiter").set(CapacityLimiter(2))
    model.eval()
    print(model)
    yield
    # Clean up the ML models and release the resources


webApp = FastAPI(lifespan=lifespan)

folder_static = f"{____workingDir}/static"
isExist_static = os.path.exists(folder_static)
if not isExist_static:
    os.makedirs(folder_static)

webApp.mount(folder_static, StaticFiles(directory=folder_static), name="static")

webApp.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conversation_sessions = {}


@webApp.post("/apis/llm/get")
async def llm_get(conversationid: str = Form(f"{uuid.uuid4()}")):
    uuid.UUID(conversationid)
    if conversationid not in conversation_sessions.keys():
        return {"ok": 0, "history": []}

    return {"ok": 1, "history": conversation_sessions[conversationid]}
    pass


# Define a chat template function
def vicuna_apply_chat_template(user_input):
    return f"<|user|>: {user_input} <|bot|>:"


# Define a custom chat template function
def gemma_apply_chat_template(messages):
    chat_history = ""
    for message in messages:
        chat_history += f"{message['role']}: {message['content']}\n"
    return chat_history

@webApp.post("/apis/llm/ask")
async def llm_ask(
    msg: str = Form(None),
    conversationid: str = Form(f"{uuid.uuid4()}"),
    systemguideline: str = Form(system_prompt),
):
    uuid.UUID(conversationid)
    conversation = []
    if conversationid not in conversation_sessions.keys():
        if systemguideline == None or systemguideline == "":
            conversation = [{"role": "system", "content": system_prompt}]
        else:
            conversation = [{"role": "system", "content": systemguideline}]

        conversation_sessions[conversationid] = conversation
    else:
        conversation = conversation_sessions[conversationid]

    t1 = datetime.datetime.now().timestamp()
    conversation.append({"role": "user", "content": msg})

    # input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(
    #     model.device
    # )
    input_ids = tokenizer(gemma_apply_chat_template(conversation), return_tensors="pt").to(model.device)
    
    out_ids = model.generate(
        input_ids=input_ids,
        # max_new_tokens=768,
        max_new_tokens=512,
        pad_token_id=2,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        temperature=0.1,
        repetition_penalty=1.05,
    )
    assistant = tokenizer.batch_decode(
        out_ids[:, input_ids.size(1) :], skip_special_tokens=True
    )[0].strip()
    conversation.append({"role": "assistant", "content": assistant})

    conversation_sessions[conversationid] = conversation
    t2 = datetime.datetime.now().timestamp()
    return {"ok": 1, "content": assistant, "elapsed": t2 - t1}

    pass


conversation_sessions_gpt_similar = {}


@webApp.post("/api/chat/get")
async def llm_chat_get(request: ConversationRequest):
    if request.conversationid in conversation_sessions_gpt_similar.keys():
        return conversation_sessions_gpt_similar[request.conversationid]
        pass

    return []


@webApp.post("/v1/embeddings")
async def llm_embedin(request: EmbeddingRequest):

    inputs = tokenizer(request.input, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]

    embedding = torch.mean(embeddings, dim=1).squeeze().numpy().tolist()
        
        # Create an Embedding instance
    embedding_instance = Embedding(
        object="embedding",
        embedding=embedding,
        index=0
    )

    # Create a Usage instance
    usage_instance = Usage(
        prompt_tokens=0,
        total_tokens=0
    )


        # Create an EmbeddingResponse instance
    embedding_response = EmbeddingResponse(
        object="list",
        data=[embedding_instance],
        model="vistral-7b",
        usage=usage_instance
    )


    return embedding_response
    pass


@webApp.post("/v1/chat/completions")
async def llm_chat_completion(request: ChatRequest):
    print("request-----------B")
    print(request)
    print("request-----------E")
    t1 = datetime.datetime.now().timestamp() * 1000

    if request.conversationid not in conversation_sessions_gpt_similar.keys():
        conversation_sessions_gpt_similar[request.conversationid] = request
        pass

    # input_ids = tokenizer.apply_chat_template(request.messages, return_tensors="pt").to(
    #     model.device
    # )
    
    input_ids = tokenizer(gemma_apply_chat_template(conversation),return_tensors="pt").to(model.device)
    
    out_ids = model.generate(
        **input_ids,
        # max_new_tokens=768,
        max_new_tokens=request.max_tokens,
        pad_token_id=2,
        do_sample=request.do_sample,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        repetition_penalty=request.repetition_penalty,
    )
    # assistants = tokenizer.batch_decode(
    #     out_ids[:, input_ids.size(1) :], skip_special_tokens=request.skip_special_tokens
    # )
    assistants = tokenizer.batch_decode( out_ids ,skip_special_tokens=request.skip_special_tokens)
    
    # print(assistant)

    choices = []
    for idx, assistant in enumerate(assistants):
        print(request.requestid)
        print(assistant)
        msg = Message(role="assistant", content=assistant.strip())

        request.messages.append(msg)
        choice = Choice(message=msg, finish_reason="stop", index=idx)
        choices.append(choice)

    conversation_sessions_gpt_similar[request.conversationid] = request

    response_id = str(uuid.uuid4())
    response_object = "chat.completion"
    response_created = int(datetime.datetime.utcnow().timestamp())

    t2 = datetime.datetime.now().timestamp() * 1000

    chat_response = ChatResponse(
        id=response_id,
        object=response_object,
        created=response_created,
        model=request.model,
        choices=choices,
        conversationid=request.conversationid,
        tools=None,
        requestid=request.requestid,
        elapsed_in_milis=t2 - t1,
    )

    return chat_response

    pass


@webApp.get("/")
async def root():
    return RedirectResponse("/docs")
    return "swagger API docs: /docs"


def runUvicorn(port):
    uvicorn.run(webApp, host="0.0.0.0", port=int(port), log_level="info")


if __name__ == "__main__":
    runUvicorn(_http_port)

# pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.7

# while True:
#     human = input("Human: ")
#     if human =="":
#         human=msg
#     if human.lower() == "reset":
#         conversation = [{"role": "system", "content": system_prompt }]
#         print("The chat history has been cleared!")
#         continue

#     conversation.append({"role": "user", "content": human })
#     input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)

#     out_ids = model.generate(
#         input_ids=input_ids,
#         max_new_tokens=768,
#         pad_token_id=2,
#         do_sample=True,
#         top_p=0.95,
#         top_k=40,
#         temperature=0.1,
#         repetition_penalty=1.05,
#     )
#     assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
#     print("Assistant: ", assistant)
#     conversation.append({"role": "assistant", "content": assistant })


"""
You need to extract the criteria export code and conversation topic code as JSON format with the following structure:{\"Criterias\":[{\"Code\":\"agent-self-introduce\",\"Name\": \"Điện thoại viên chào hỏi khách hàng\", \"Value\": the content that agent introduce herself,\"Times\":array of all the start times in seccond when agent introduce herself },{\"Code\":\"customer-name\",\"Name\": \"Tên khách hàng\", \"Value\": customer name,\"Times\":array of all the start times in seccond when agent use customer name},{\"Code\":\"customer-name\",\"Name\": \"Tên khách hàng\", \"Value\": customer name,\"Times\":array of all the start times in seccond when agent use customer name},{\"Code\":\"customer-name\",\"Name\": \"Tên khách hàng\", \"Value\": customer name,\"Times\":array of all the start times in second when agent use customer name},{\"Code\":\"customer-complain-agent-bad-attitude\",\"Name\": \"Khách hàng phàn nàn thái độ điện thoại viên\", \"Value\": customer complain that agent has bad attitude,\"Times\":array of all the start times in second when customer complain that agent has bad attitude},{\"Code\":\"customer-complain-agent-voice\",\"Name\": \"Khách hàng phàn nàn âm lượng điện thoại viên\", \"Value\": customer complain that agent voice is difficult to hear,\"Times\":aray of all the start times in second when customer complain that agent voice is difficult to hear},{\"Code\":\"password-info\",\"Name\": \"Thông tin mật khẩu\", \"Value\": customer secret password info,\"Times\":aray of all the start times in second when customer secret password info is found}],\"Topic\": conversation topic } In there,\n- Criterias is a list of criteria. Each criterion includes:\n\t+ Code is the criteria code.\n\t+ Name is the criteria name.\n\t+ Text is entities extracted\n\t+ Times array of all the start times in seccond when the saying appears in conversation\n- Topic is the topic of the conversation\n\nThis is 4 topics codes include:\n- Nghiệp vụ thẻ, included keywords: đóng thẻ,mở thẻ,phát hành,câu hỏi bảo mật,hạn mức - Nghiệp vụ vay, included keywords: khoản vay,hạn thanh toán,ngày trả nợ,thu nợ,quét nợ,mở vay,ngày vay,ngày trả nợ - Nghiệp vụ NEO, included keywords: mật khẩu,tên đăng nhập,lỗi chuyển tiền,không đăng nhập - Nghiệp vụ tiết kiệm, included keywords: lãi suất,sổ,số tiền sổ,ngày gửi,tất toán sổ,giải toả sổ,số sổ\n\nYou should only respond in a JSON structure. If any criteria or topics cannot be extracted, return \"\". Get as much information as possible\n\nThis is a conversation that needs to be handle\nNote: Number inside () is start time\nAgent(2.319999933242798): dạ danh em ly xin nghe Agent(3.7899999618530273): em có thể hỗ trợ anh chị thông tin gì ạ Customer(5.800000190734863): ừ em ơi Customer(6.679999828338623): chị có một cái thẻ tín dụng ờ classic nó mobifone ý Customer(9.889999389648438): mà chị đang tìm hiểu về có nhu cầu muốn nâng lên cái mở cái thẻ ly ly được không em Agent(14.769999504089355): ờ chị vui lòng cho em xin lại tên của chị để em tiện trao đổi ạ Customer(18.489999771118164): ờ chị liên liên em ạ Agent(20.169998168945312): vâng Agent(20.8700008392334): chị cho em xin lại chứng minh thư để em kiểm tra thêm thông tin ạ Customer(24.10999870300293): ờ đúng rồi chờ chị một chút nhá Customer(26.159997940063477): chứng minh thư là Agent(26.409997940063477): dạ vâng Customer(27.729999542236328): 123 Customer(28.90999984741211): 456789 em ơi Agent(31.23999786376953): dạ vâng Agent(32.09000015258789): chị vui lòng đợi máy giúp em trong giây lát nhá Customer(34.59000015258789): ok Customer(58.269996643066406): nó bảo phải chờ máy Customer(70.37000274658203): biết được 30 giây Customer(75.91999816894531): alo em ơi được chưa em ơi Agent(78.41000366210938): à dạ cảm ơn chị chờ máy xin lỗi để chị chờ máy hơi lâu ạ Agent(81.41000366210938): ờ đối với trường hợp của Agent(83.22999572753906): chị hiên thì Agent(84.60999298095703): hiện tại em kiểm tra thông tin thì chị nạp chưa nằm trong danh sách được mở sim thẻ Agent(88.18999481201172): đây là rất tiếc bên em chưa thể hỗ trợ mình mở thêm thẻ rồi chị ạ Customer(91.9000015258789): thế hả Agent(93.05999755859375): dạ vâng Customer(94.61000061035156): mình cần trách danh sách hả em Agent(96.41999816894531): dạ đúng rồi nếu trong trường hợp này mình có nhu cầu sử dụng ờ Agent(100): cái thẻ Agent(100.75): setup ý ạ Agent(101.41000366210938): thì chị có thể đóng lại cái thẻ classic mobifone giúp em Agent(104.75): và vui lòng chờ giúp em sau 30 ngày kể từ thời điểm đóng thẻ Agent(107.47999572753906): hợp đồng đóng thẻ của mình sẽ trình hồ sơ để mở thêm thẻ giáp bát chị nhé Customer(112.12000274658203): ừ Customer(113.02999114990234): tức là có nghĩa bây giờ chị phải đóng cái thẻ của chị đang xài Agent(115.56999969482422): dạ đúng rồi Customer(115.77999877929688): sau đó rồi chị 30 ngày sau thì chị mới mở được cái dây thẻ mới Agent(119.86000061035156): dạ đúng rồi ạ Agent(119.96000061035156): chị cho em xin số otp gửi về Customer(120.78999938964844): 76342///Customer(120.88999938964844): chỉ mỗi cách này thôi hả em Agent(122.43000030517578): dạ đúng rồi ạ chỉ có mỗi cách này thôi ạ bởi vì hiện tại em kiểm tra thông tin thì rất tiếc là chị không nằm trong danh sách mở thêm thẻ bên em Customer(129.2899932861328): sắp hai nhiều khách hàng có nhu cầu muốn sử dụng thêm thì phải Customer(132.6199951171875): trong danh sách mới mở được à Agent(134.1999969482422): dạ vâng đúng rồi ạ Agent(136.36000061035156): thì không biết là hiện tại chị có muốn đóng thẻ classic không ạ Customer(136.7100067138672): ừ Customer(139.5800018310547): thôi để chị xem Customer(141.55999755859375): em thêm nhá chị tìm hiểu đã Customer(142.97000122070312): chị nắm được thông tin rồi Agent(144.6999969482422): dạ vâng không biết là chị hiên còn thắc mắc thông tin nào khác nữa không chị Customer(147.80999755859375): ừ thôi chị không có thông tin gì nữa đâu Customer(149.84999084472656): chị cảm ơn Agent(150.58999633789062): dạ vâng thế có vấn đề gì thì yên vui lòng liên hệ lại sau giúp em ạ Agent(153.45999145507812): cảm ơn chị em chào chị hen Customer(154.69000244140625): ok 
"""
