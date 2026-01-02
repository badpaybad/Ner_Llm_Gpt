
import gpt2helper

gpt2_model, gpt2_tokenizer= gpt2helper.build_joke_model_tokenzier_predictor(
    model_name="gpt2_medium_joker_4.pt" ,
                            models_folder="/work/Ner_Llm_Gpt/gpt2usage/gpt2_trained_models"
    )
import datetime

print(datetime.datetime.now())
gpt2helper.generate_some_text_joke(gpt2_model, gpt2_tokenizer,"Make a joke about Elon Musk")
print(datetime.datetime.now())

print(datetime.datetime.now())
gpt2helper.generate_some_text_joke(gpt2_model, gpt2_tokenizer,"Make a joke about Elon Musk")
print(datetime.datetime.now())