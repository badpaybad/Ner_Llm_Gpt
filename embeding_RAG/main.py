from transformers import LlamaTokenizer, LlamaModel, Qwen2Model, Qwen2Tokenizer
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

modelpath = "/mldlai/Qwen1.5-4B-Chat"

def llama():
    # Load the tokenizer and model
    tokenizer = LlamaTokenizer.from_pretrained(modelpath)
    model = LlamaModel.from_pretrained(
        modelpath,
        device_map="cpu",
        use_cache=False,
    )

    # Encode the text
    text = "Hello, Hugging Face!"
    inputs = tokenizer(text, return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    # Extract the embeddings for the [CLS] token or average all token embeddings
    # Here, we average all token embeddings
    embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

    print("llama Embedding shape:", embedding.shape)
    print("Embedding:", embedding)
    


def transfromAutoMl():

    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForCausalLM.from_pretrained(
        modelpath,
        #torch_dtype=torch.bfloat16,  # change to torch.float16 if you're using V100
        device_map="cpu",
        use_cache=False,
    )

    inputs = tokenizer("Hello, Hugging Face!", return_tensors="pt")

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]

    embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

    print("transfromAutoMl Embedding shape:", embedding.shape)
    print("Embedding:", embedding)
    pass


# llama()
transfromAutoMl()


# # Load the tokenizer and model
# model_name = "gpt2"  # Replace with the specific model you want to use, e.g., "facebook/llama-7b" or "Qwen/Qwen-7b"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# # Convert model parameters to float32 if needed
# model = model.to(torch.float32)

# # Encode the text
# text = "Hello, Hugging Face!"
# inputs = tokenizer(text, return_tensors='pt')

# # Convert inputs to float32
# inputs = {k: v.to(torch.float32) for k, v in inputs.items()}

# # Generate embeddings
# with torch.no_grad():
#     outputs = model(**inputs, output_hidden_states=True)
#     hidden_states = outputs.hidden_states

# # Use the last hidden state as embeddings
# embeddings = hidden_states[-1].to(torch.float32)

# # Average the embeddings of all tokens
# embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

# print("Embedding shape:", embedding.shape)
# print("Embedding:", embedding)