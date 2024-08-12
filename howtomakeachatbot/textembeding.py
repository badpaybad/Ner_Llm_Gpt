import time
from transformers import LlamaTokenizer, LlamaModel, Qwen2Model, Qwen2Tokenizer
import torch
import numpy
from numpy import dot
from numpy.linalg import norm

from transformers import AutoModelForCausalLM, AutoTokenizer


class TextEmbeding:
    def __init__(self):
        self.modelpath = "/work/shared/vicuna-7b-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelpath)
        self. model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=self.modelpath,
            # torch_dtype=torch.bfloat16,  # change to torch.float16 if you're using V100
            device_map="cpu",  # cuda
            use_cache=False,
        )
        print(self.modelpath)
        pass

    def getTextVector(self, text):

        inputs = self.tokenizer(text, return_tensors="pt")

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1]

        embedding = torch.mean(embeddings, dim=1).squeeze().numpy()

        # print("transfromAutoMl Embedding shape:", embedding.shape)
        # print("Embedding:", embedding)
        return embedding
        pass

    def consineSimilar(self, a, b):

        cos_sim = numpy.dot(a, b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b))
        return cos_sim

    def _dot(self, A, B):
        return (sum(a*b for a, b in zip(A, B)))

    def _cosine_similarity(self, a, b):
        return self._dot(a, b) / ((self._dot(a, a) ** .5) * (self._dot(b, b) ** .5))


test = TextEmbeding()
res = test.getTextVector("hi du")
print(len(res))

time.sleep(1)

res1 = test.getTextVector("hi dung")
print(res1)

print(f"consine similar: {test.consineSimilar(res,res1)}")
print(f"consine similar1: {test._cosine_similarity(res,res1)}")
