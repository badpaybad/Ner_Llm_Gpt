
import gpt2helper

gpt2_model, gpt2_tokenizer= gpt2helper.build_pretrain_model_tokenizer()

joke_loader = gpt2helper.build_joker_loader()

gpt2helper.gpt2_train( gpt2_model,gpt2_tokenizer,joke_loader=joke_loader)

gpt2helper.gpt2_predict_test(gpt2_model, gpt2_tokenizer, model_epoch_name="gpt2_medium_joker_4.pt" ,
                            models_folder="/work/Ner_Llm_Gpt/gpt2usage/gpt2_trained_models")


# after train and got model into folder gpt2_trained_models 
# run test.py to generate from input text