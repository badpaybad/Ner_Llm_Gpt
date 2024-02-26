
import gpt2helper

gpt2_model, gpt2_tokenizer= gpt2helper.build_pretrain_model_tokenizer()

joke_loader = gpt2helper.build_joker_loader()

gpt2helper.gpt2_predict_test(gpt2_model, gpt2_tokenizer,gpt2helper.MODEL_EPOCH)
