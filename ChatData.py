from torch.utils.data import Dataset
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer



class ChatData(Dataset):
    def __init__(self, path:str, tokenizer):
        self.data = json.load(open(path, 'r'))

        self.X = []
        for i in self.data:
            parts = i['input'].split('[|AI|]')
            human_texts = parts[0].split('[|Human|]')[1]  # Extract only the part after '|Human|'
            ai_texts = (parts[1] if len(parts) > 1 else "").replace('[|Human|]','')  # Extract only the part after '|AI|'
            human_texts = human_texts.replace('\n','')
            ai_texts = ai_texts.replace('\n','')
            self.X.append('<|startoftext|> '+human_texts+' <bot>: '+ai_texts+' <|endoftext|>')
        
        # print(self.X[:10])
        # X has a format: "<startoftext> hello how are you <bot>: i am good <endoftext>"
        # self.X = self.X[:int(1e3)]
        self.X_encoded = tokenizer(self.X,padding=True,truncation = True, return_tensors="pt",max_length=512)
        self.input_ids = self.X_encoded['input_ids']
        self.attention_mask = self.X_encoded['attention_mask']
        # print(self.input_ids[:10])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx])



if __name__ == "__main__":
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.add_tokens(["<bot>:"])
    chat = ChatData("./alpaca_chat_data.json",tokenizer)