from transformers import LlamaTokenizer, LlamaModel, Qwen2Model, Qwen2Tokenizer
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

class TextEmbeding:
    def __init__(self) :
        self.modelpath="/work/llm/Ner_Llm_Gpt/mistralvn/Vistral-7B-Chat"
        pass
        

    def getTextVector(self,text):
        print(self.modelpath)
        tokenizer = AutoTokenizer.from_pretrained( self.modelpath)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path= self.modelpath,
            #torch_dtype=torch.bfloat16,  # change to torch.float16 if you're using V100
            device_map="cpu",# cuda
            use_cache=False,
        )

        inputs = tokenizer(text, return_tensors="pt")

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]

        embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

        # print("transfromAutoMl Embedding shape:", embedding.shape)
        # print("Embedding:", embedding)
        return embedding
        pass

test= TextEmbeding()
res=test.getTextVector("hi du")
print (res)