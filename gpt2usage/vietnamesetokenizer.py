import underthesea 
from transformers import GPT2Tokenizer

class VietnameseGpt2Tokenizer():
    def __init__(self) :
        self.tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2-medium")
        pass
    
    def encode(self, text):
        #tokens_underthesea = underthesea.word_tokenize(text, format="text")
        tokens_underthesea = underthesea.word_tokenize(text)
        tokens_mapped = [token.lower() for token in tokens_underthesea]
        #print(tokens_mapped)
        encoded_input = self.tokenizer_gpt2.encode(" ".join(tokens_mapped))
        return encoded_input
        pass
    
    
    def decode(self, list_encoded):
        decoded_text = self.tokenizer_gpt2.decode(list_encoded)
        return decoded_text
        pass