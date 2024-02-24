
import gpt2helper

gpt2_model, gpt2_tokenizer= gpt2helper.build_joke_model_tokenzier_predictor("gpt2_medium_joker_0_test.pt")
import datetime

print(datetime.datetime.now())
gpt2helper.generate_some_text_joke(gpt2_model, gpt2_tokenizer,"Telling my daughter garlic is good for you")
print(datetime.datetime.now())