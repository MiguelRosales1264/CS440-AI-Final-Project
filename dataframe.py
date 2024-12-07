import pandas as pd

df = pd.read_parquet("hf://datasets/dexaai/huberman_on_exercise/data/data-00000-of-00001-8e5e40fbf9236004.parquet")
# print(df.head())

# Remove Second Column (Embedding)
df = df.drop(columns=['embedding'])

import re
# Remove punctuation (keeps numbers and capitalization)
# df['document'] = df['document'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
# print(df.head())

# Convert to list in order to combine all documents into one list
corpus = df['document'].tolist()
# print(type(corpus))
# print(corpus[:5])

# Convert list to string by joining all documents
text = " ".join(corpus)
# print(text[:1000])
print(len(text)) # The number of chars in the corpus
# print(type(text))

# Create a list of all unique characters in the text
vocabulary = sorted(list(set(text)))
vocabulary_size = len(vocabulary)

# print('Vocabulary Size: ', vocabulary_size) # The number of unique characters in the text
# print('Vocabulary in text: ', ''.join(vocabulary))

import tiktoken
# print(tiktoken.list_encoding_names())
encoder = tiktoken.get_encoding('o200k_base')
# assert encoder.decode(encoder.encode("hello world")) == "hello world"
# print(encoder.decode(encoder.encode("hello world")))

# Map each character to an index
stoi = {c: i for i, c in enumerate(vocabulary)}
itos = {i: c for i, c in enumerate(vocabulary)}
encode = lambda x: [stoi[c] for c in x] # encode('a') -> 0, takes a character and returns its index in the vocabulary
decode = lambda s: ''.join(itos[i] for i in s) # decode([0, 1, 2]) -> 'abc', takes a list of indices and returns the corresponding string
# print('Encoded:', encode("andrew huberman"))
# print('Decoded:', decode(encode("andrew huberman")))

# Encode the entire text with torch
import torch

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.type())
# print(data[:1000])

# Split into training and validation sets
train_size = int(0.9 * len(data))
val_data = data[train_size:]
train_data = data[:train_size]
# print('Train size:', len(train_data))
# print('Val size:', len(val_data))

# block_size = 8
# train_data[:block_size]


x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t + 1]
    target = y[t]
    # print(f"when input is {context} the target is {target}")

torch.manual_seed(47)
batch_size = 4 # number of sequences in a batch, processed in parallel
block_size = 8 # length of a sequence, max context length for predictions

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

xb, yb = get_batch('train') 
# print('Inputs: ', xb.shape)
# print(xb)
# print('Targets: ', yb.shape)
# print(yb)

# print('------')


for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t + 1]
        target = yb[b, t]
        # print(f"when input is {context.tolist()} the target is {target}")
    
print(xb)

import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, vocabulary_size)
        
    def forward(self, idx, targets=None):
        
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


model = BigramLanguageModel(vocabulary_size)
logits, loss = model(xb, yb)
print(logits.shape)
print(loss) # -ln(1/vocabulary_size) = -ln(1/76) = 4.3307

print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

# HAVE NOT ADDED GUI YET
