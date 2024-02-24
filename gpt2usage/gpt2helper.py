from torch.utils.data import Dataset
from transformers import AdamW, get_linear_schedule_with_warmup
# from transformers import AdamW, WarmupLinearSchedule
import csv
import json
import os
from torch.utils.data import Dataset, DataLoader
import warnings
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np

import logging
logging.getLogger().setLevel(logging.CRITICAL)

warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print("pytorch with device: ")
print(device)

MODELS_FOLDER = "gpt2_trained_models"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 3e-5  # 5e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400

MODEL_EPOCH = 4


def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # Normalize
    choice = np.random.choice(n, 1, p=top_prob)
    token_id = ind[choice][0]
    return int(token_id)


class JokesDataset(Dataset):
    def __init__(self, jokes_dataset_path='/home/dunp/Downloads/archive/'):
        super().__init__()

        short_jokes_path = os.path.join(jokes_dataset_path, 'shortjokes.csv')
        '''
        "ID","Joke"
        1,"[me narrating a documentary about narrators] ""I can't hear what they're saying cuz I'm talking"""
        2,"Telling my daughter garlic is good for you. Good immune system and keeps pests away.Ticks, mosquitos, vampires... men."
        3,"I've been going through a really rough period at work this week It's my own fault for swapping my tampax for sand paper."
        4,"If I could have dinner with anyone, dead or alive... ...I would choose alive. -B.J. Novak-"
        '''

        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"

        with open(short_jokes_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            x = 0
            for row in csv_reader:
                joke_str = f"JOKE:{row[1]}{self.end_of_text_token}"
                self.joke_list.append(joke_str)

    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]


# from transformers import AdamW, WarmupLinearSchedule
# pip install pytorch_pretrained_bert
# pip install --upgrade transformers


def gpt2_train(model, tokenizer, joke_loader=DataLoader(JokesDataset(), batch_size=BATCH_SIZE, shuffle=True),models_folder=MODELS_FOLDER):

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    model = model.to(device)
    model.train()

    # optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=WARMUP_STEPS, t_total = -1)

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(joke_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    proc_seq_count = 0
    sum_loss = 0.0
    batch_count = 0

    tmp_jokes_tens = None

    if not os.path.exists(models_folder):
        os.mkdir(models_folder)

    for epoch in range(EPOCHS):

        print(f"EPOCH {epoch} started" + '=' * 30)

        for idx, joke in enumerate(joke_loader):

            #################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
            joke_tens = torch.tensor(tokenizer.encode(
                joke[0])).unsqueeze(0).to(device)
            # Skip sample from dataset if it is longer than MAX_SEQ_LEN
            if joke_tens.size()[1] > MAX_SEQ_LEN:
                continue

            # The first joke sequence in the sequence
            if not torch.is_tensor(tmp_jokes_tens):
                tmp_jokes_tens = joke_tens
                continue
            else:
                # The next joke does not fit in so we process the sequence and leave the last joke
                # as the start for next sequence
                if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > MAX_SEQ_LEN:
                    work_jokes_tens = tmp_jokes_tens
                    tmp_jokes_tens = joke_tens
                else:
                    # Add the joke to sequence, continue and try to add more
                    tmp_jokes_tens = torch.cat(
                        [tmp_jokes_tens, joke_tens[:, 1:]], dim=1)
                    continue
            ################## Sequence ready, process it trough the model ##################

            outputs = model(work_jokes_tens, labels=work_jokes_tens)
            loss, logits = outputs[:2]
            loss.backward()
            sum_loss = sum_loss + loss.detach().data

            proc_seq_count = proc_seq_count + 1
            if proc_seq_count == BATCH_SIZE:
                proc_seq_count = 0
                batch_count += 1
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            if batch_count == 100:
                print(f"sum loss {sum_loss}")
                batch_count = 0
                sum_loss = 0.0

        # Store the model after each epoch to compare the performance of them
        torch.save(model.state_dict(), os.path.join(
            models_folder, f"gpt2_medium_joker_{epoch}.pt"))



def gpt2_predict_test(model, tokenizer, model_epoch=MODEL_EPOCH, models_folder=MODELS_FOLDER):

    model_path = os.path.join(
        models_folder, f"gpt2_medium_joker_{model_epoch}.pt")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    jokes_output_file_path = f'generated_{model_epoch}.jokes'
    if os.path.exists(jokes_output_file_path):
        os.remove(jokes_output_file_path)

    joke_num = 0
    with torch.no_grad():

        for joke_idx in range(1000):

            joke_finished = False

            cur_ids = torch.tensor(tokenizer.encode(
                "JOKE:")).unsqueeze(0).to(device)

            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                # Take the first(from only one in this case) batch and the last predicted embedding
                softmax_logits = torch.softmax(logits[0, -1], dim=0)
                if i < 3:
                    n = 20
                else:
                    n = 3
                # Randomly(from the topN probability distribution) select the next word
                next_token_id = choose_from_top(
                    softmax_logits.to('cpu').numpy(), n=n)
                cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(
                    device) * next_token_id], dim=1)  # Add the last word to the running sequence

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    joke_finished = True
                    break

            if joke_finished:

                joke_num = joke_num + 1

                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)

                with open(jokes_output_file_path, 'a') as f:
                    f.write(f"{output_text} \n\n")


def generate_some_text_joke(model, tokenizer, input_str, text_len=250):
    
    print("generate_some_text_joke:BEGIN")

    cur_ids = torch.tensor(tokenizer.encode(
        "JOKE:"+input_str)).unsqueeze(0).long().to(device)

    model.eval()
    with torch.no_grad():

        for i in range(text_len):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            # Take the first(only one) batch and the last predicted embedding
            softmax_logits = torch.softmax(logits[0, -1], dim=0)
            # Randomly(from the given probability distribution) choose the next word from the top n words
            next_token_id = choose_from_top(
                softmax_logits.to('cpu').numpy(), n=10)
            cur_ids = torch.cat([cur_ids, torch.ones((1, 1)).long().to(
                device) * next_token_id], dim=1)  # Add the last word

        output_list = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_list)
        print(output_text)
        print("generate_some_text_joke:END")
        return output_text


def build_joker_loader():
    dataset = JokesDataset()
    joke_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return joke_loader


def build_pretrain_model_tokenizer():

    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    gpt2_model = gpt2_model.to(device)

    return gpt2_model, gpt2_tokenizer


def build_joke_model_tokenzier_predictor(model_name=f"gpt2_medium_joker_{MODEL_EPOCH}.pt", models_folder=MODELS_FOLDER):
    gpt2_model, gpt2_tokenizer = build_pretrain_model_tokenizer()

    model_path = os.path.join(
        models_folder, model_name)
    gpt2_model.load_state_dict(torch.load(model_path))
    gpt2_model.eval()
    return gpt2_model,gpt2_tokenizer
