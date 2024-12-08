import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 32 # number of sequences in a batch, processed in parallel
block_size = 8 # length of a sequence, max context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32  # 32 dimensional embedding. This is the size of the "input" to each transformer layer
# head_size = 16
seed = 47

# Import the data
df = pd.read_parquet("hf://datasets/dexaai/huberman_on_exercise/data/data-00000-of-00001-8e5e40fbf9236004.parquet")

# Remove Second Column (Embedding)
df = df.drop(columns=['embedding'])

# Convert to list in order to combine all documents into one list
corpus = df['document'].tolist()

# Convert list to string by joining all documents
text = " ".join(corpus)

# Create a list of all unique characters in the text
vocabulary = sorted(list(set(text)))
vocabulary_size = len(vocabulary)

# print('Vocabulary Size: ', vocabulary_size)  # The number of unique characters in the text
# print('Vocabulary in text: ', ''.join(vocabulary))  # The unique characters in the text


# Map each character to an index
stoi = {c: i for i, c in enumerate(vocabulary)}
itos = {i: c for i, c in enumerate(vocabulary)}
encode = lambda x: [stoi[c] for c in x] # encode('a') -> 0, takes a character and returns its index in the vocabulary
decode = lambda s: ''.join(itos[i] for i in s) # decode([0, 1, 2]) -> 'abc', takes a list of indices and returns the corresponding string

# Encode the entire text with torch
data = torch.tensor(encode(text), dtype=torch.long)

# Split into training and validation sets
train_size = int(0.9 * len(data))
val_data = data[train_size:]
train_data = data[:train_size]

x = train_data[:block_size]
y = train_data[1:block_size+1]

torch.manual_seed(seed)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    output = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        output[split] = losses.mean()
    model.train()
    return output

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.val = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x, head_size):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # compute the attention weights
        weight = q @ k.transpose(-2, -1) / (head_size ** 0.5)  # (B, T, C) @ (B, C, T) --> (B, T, T)
        weight = weight.masked_fill(self.tril == 0, float('-inf'))
        weight = F.softmax(weight, dim=1)
        
        v = self.val(x) # (B, T, C)
        output = weight @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return output
        
        
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # each position maps to a learnable embedding
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocabulary_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        tok_embd = self.token_embedding_table(idx)  # (B, T, C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, C)
        x = tok_embd + pos_embd  # (B, T, C)
        x = self.sa_head(x)  # (B, T, C), applies one head of self-attention
        logits = self.lm_head(x)  # (B, T, vocabulary_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)  # batch, time, channels
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) 
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last token/time step
            logits = logits[:, -1, :]
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses =  ()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_batch('train')
        
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))


# HAVE NOT ADDED GUI YET
