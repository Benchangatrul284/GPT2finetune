from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ChatData import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os

def infer(inp):
    inp = "<|startoftext|> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a,pad_token_id=tokenizer.eos_token_id,max_new_tokens=100)
    output = tokenizer.decode(output[0])
    return output.split('<bot>:')[1].strip().split('<|endoftext|>')[0].replace('<|endoftext|>','')



if __name__ == '__main__':
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    # set tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                    # "bos_token": "<startoftext>",
                                    # "eos_token": "<endoftext>"
                                    })
    tokenizer.add_tokens(["<bot>:"])

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(os.path.join("/local_data/U114_llama_2/checkpoints","model_state.pt")))
    model = model.to(device)
    model.eval()
    print("Welcome to ChatBot")
    while True:
        inp = input('User:')
        print(infer(inp))