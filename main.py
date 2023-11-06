from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ChatData import ChatData
from torch.optim import Adam
from torch.utils.data import DataLoader
import tqdm
import torch
from tqdm import tqdm
import os
import torch.cuda.amp as amp

def train(chatData, model, optim):

    epochs = 50
    scaler = amp.GradScaler()
    for i in (range(epochs)):
        print("epoch : ", i)
        for X, a in tqdm(chatData):
            X = X.to(device)
            a = a.to(device)
            optim.zero_grad()
            with amp.autocast():
                loss = model(X, attention_mask=a, labels=X).loss
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

        if i%1==0:
            print(infer("Hello, how are you doing?"))
            torch.save(model.state_dict(), os.path.join("/local_data/U114_llama_2/checkpoints","model_state.pt"))


def infer(inp):
    inp = "<|startoftext|> "+inp+" <bot>: "
    inp = tokenizer(inp, return_tensors="pt")
    X = inp["input_ids"].to(device)
    a = inp["attention_mask"].to(device)
    output = model.generate(X, attention_mask=a,pad_token_id=tokenizer.eos_token_id,max_new_tokens=100)
    output = tokenizer.decode(output[0])
    return output

if __name__ == "__main__":
    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # set tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<pad>", 
                                   #"bos_token": "<startoftext>",
                                   #"eos_token": "<endoftext>"
                                 })
    
    tokenizer.add_tokens(["<bot>:"])

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    # print(tokenizer.decode(model.generate(**tokenizer("hey i was good at basketball but ",
    #                          return_tensors="pt"))[0]))

    chatData = ChatData("./alpaca_chat_data.json", tokenizer)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer,padding=True,return_tensors='pt')
    chatData =  DataLoader(chatData, batch_size=16, shuffle=True, num_workers=4)
    model.train()
    print("training .... ")
    train(chatData, model, optim = Adam(model.parameters(), lr=1e-3))

    model.eval()
    print("infer from model : ")
    while True:
        inp = input()
        print(infer(inp))