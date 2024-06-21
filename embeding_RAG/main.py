from transformers import LlamaTokenizer, LlamaModel
import torch

modelpath="/mldlai/Vistral-7B-Chat"
# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(modelpath)
model = LlamaModel.from_pretrained(modelpath)

# Encode the text
text = "Hello, Hugging Face!"
inputs = tokenizer(text, return_tensors='pt')

# Generate embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state

# Extract the embeddings for the [CLS] token or average all token embeddings
# Here, we average all token embeddings
embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

print("Embedding shape:", embedding.shape)
print("Embedding:", embedding)
