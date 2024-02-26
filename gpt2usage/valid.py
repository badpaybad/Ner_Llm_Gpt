
import gpt2helper

gpt2_model, gpt2_tokenizer= gpt2helper.build_pretrain_model_tokenizer()

gpt2helper.gpt2_predict_test(gpt2_model, gpt2_tokenizer,"gpt2_medium_joker_4.pt",gpt2helper.MODELS_FOLDER)
