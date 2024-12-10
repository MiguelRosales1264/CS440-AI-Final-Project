import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset

batch_size = 32 # number of sequences in a batch, processed in parallel
block_size = 32 # length of a sequence, max context length for predictions (time steps)
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32 # 32 dimensional embedding. This is the size of the "input" to each transformer layer
# head_size = 16
seed = 47
n_layers = 4  # --> 4 transformer layers (num blocks)
n_heads = 4   # --> 4 attention heads in each transformer layer
dropout = 0.2 # --> prevents overfitting by randomly 'dropping out' some of the data (cuts communication between tokens)

torch.manual_seed(seed)

# NOTE: python3.9 -m pip install datasets
# if you run into any issues, use the above commnad
# Import the data
df = load_dataset("dexaai/huberman_on_exercise")
# print(df)

# Conver to pands df
df = pd.DataFrame(df['document'])
df = df.to_pandas()

# Remove Second Column (Embedding)
# df = df.drop(columns=['embedding'])

# Convert to list in order to combine all documents into one list
corpus = df.tolist()

# Convert list to string by joining all documents
text = ' '.join(corpus)

# Create a list of all unique characters in the text
vocab = sorted(list(set(text)))
v_size = len(vocab)

# Map each character to an index
ctoi = {c: i for i, c in enumerate(vocab)}
itoc = {i: c for i, c in enumerate(vocab)}
encode = lambda x: [ctoi[c] for c in x] # encode('a') -> 0, takes a character and returns its index in the vocabulary
decode = lambda s: ''.join(itoc[i] for i in s) # decode([0, 1, 2]) -> 'abc', takes a list of indices and returns the corresponding string

# Encode the entire text with torch
data = torch.tensor(encode(text), dtype=torch.long)

# Split into training and validation sets
train_size = int(0.9 * len(data))
train_data = data[:train_size]
val_data = data[train_size:]

# our data loader function
def get_batch(split):
    # data array encoded as integers
    data = train_data if split == 'train' else val_data
    
    # This gives 'batch_size' number of random indices to start each sequence
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    
    # This stacks the sequences in a batch, (batch_size, block_size)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Averages loss over multiple batches
# Why do this? Less noisy and more stable gradients. Meaning the model will learn better
@torch.no_grad()  # -> let's PyTorch know that we don't need to keep track of gradients (back propagation) for this function
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
        

# Single Head Attention
class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.val = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, is_masked=False):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # compute the attention weights
        weight = q @ k.transpose(-2, -1) / (C ** 0.5)  # (B, T, head_size) @ (B, head_size, T) --> (B, T, T)
        if is_masked:
            weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # --> prevents attention to future tokens
        weight = F.softmax(weight, dim=-1)  # (B, T, T)  # --> normalizes the weights
        weight = self.dropout(weight)
        v = self.val(x) # (B, T, head_size)
        # print(weight.shape, v.shape)
        output = weight @ v # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)
        return output  # --> (B, T, head_size)

# CrossAttention
class CrossAttention(nn.Module):
    def __init__(self, head_size, key, val):
        super().__init()
        self.key = key
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.val = val

    def forward(self, x, is_masked=False):
        B, T, C = x.shape
        
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # compute the attention weights
        weight = q @ k.transpose(-2, -1) / (C ** 0.5)
        if is_masked:
            weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weight = F.softmax(weight, dim=-1)
        
        v = self.val(x)
        output = weight @ v
        return output    
    
# Multi-Head Attention Model
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, is_cross_attention=False):
        super().__init__()
        if is_cross_attention:
            self.heads = nn.ModuleList([CrossAttention(head_size) for _ in range(num_heads)])  # --> creates a list of heads
        else:
            self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])  # --> creates a list of heads
        self.proj = nn.Linear(n_embd, n_embd)  # --> projects the concatenated heads to the original dimension
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, is_masked=False):
        output = torch.cat([h(x, is_masked) for h in self.heads], dim=-1)  # --> concatenates the output of each head, (B, T, num_heads * head_size)
        output = self.dropout(self.proj(output))  # --> linear transformation of the concatenated heads
        return output  # --> (B, T, C)
    
# Feed Forward Neural Network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # Projection layer going back into residual pathway
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
# Decoder Block
class DecodeBlock(nn.Module):
    def __init__(self, n_embd, n_heads, key=None, val=None):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        if((key is not None) and (val is not None)):
            self.ca = CrossAttention(n_heads, head_size, key, val)
            self.ln3 = nn.LayerNorm(n_embd)  # --> for cross attention
        else:
            self.ca = None
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)  
        self.key = key
        self.val = val

    def forward(self, x, is_masked=False):
        # Residual connections (bc of the skip connection, the add) w/ layer normalization
        x = x + self.sa(self.ln1(x), is_masked=True)
        if self.ca is not None:
            x = x + self.ca(self.ln3(x), self.key, self.val)  # --> for cross attention (not masked)
        x = x + self.ffwd(self.ln2(x))
        return x
  
# Encoder Block  
class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # Residual connections (bc of the skip connection, the add) w/ layer normalization
        x = x + self.sa(self.ln1(x), is_masked=False) 
        x = x + self.ffwd(self.ln2(x))
        return x
    
# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, v_size, key=None, val=None):
        super().__init__()
        # Embedding table that maps from indices to vectors of size n_embd
        self.token_embedding_table = nn.Embedding(v_size, n_embd)
        self.postion_encoding_table = nn.Embedding(block_size, n_embd)  # each position maps to a learnable embedding. now tokens know their position
        
        self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_heads, key, val) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd, v_size)  # token_embd -> logits over the vocabulary

    def forward(self, idx, targets=None):
        # NOTE: idx and targets are (B, T) tensors of ints
        B, T = idx.shape
        
        tok_embd = self.token_embedding_table(idx)  # (B, T, n_embd) (batch, time, channels), where C is the embedding size
        pos_embd = self.postion_encoding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_embd + pos_embd  # (B, T, C)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.linear(x)  # (B, T, vocab_size), this is the prediction for the next token in the sequence
        
        # compute the loss -> this is to eval the quality of the predictions from the model
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            # Update dimensions of logits and targets to fit the loss function
            logits = logits.view(B * T, C)  # (B, T, C) -> (B * T, C)
            targets = targets.view(B * T)  # (B, T) -> (B * T)
            loss = F.cross_entropy(logits, targets)  # -> compares the predicted logits with the actual targets
        
        return logits, loss

    def generate(self, idx, max_gens=100):
        # idx is (B, T) 
        for _ in range(max_gens):
            idx_cond = idx[:, -block_size:]  # only last block_size tokens are used
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last token/time step
            logits = logits[:, -1, :]  # (B, T, vocab_size) -> (B, vocab_size)
            # apply softmax for probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        return idx

# ChadGPT model
class ChadGPT(nn.Module):
    def __init__(self, v_size):
        super().__init__()
        # Embedding table that maps from indices to vectors of size n_embd
        self.token_embedding_table = nn.Embedding(v_size, n_embd)
        self.postion_encoding_table = nn.Embedding(block_size, n_embd)  # each position maps to a learnable embedding. now tokens know their position
        
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_heads) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(n_embd)
        self.linear = nn.Linear(n_embd, v_size)  # token_embd -> logits over the vocabulary

    def forward(self, idx, targets=None):
        # NOTE: idx and targets are (B, T) tensors of ints
        B, T = idx.shape
        
        tok_embd = self.token_embedding_table(idx)  # (B, T, n_embd) (batch, time, channels), where C is the embedding size
        pos_embd = self.postion_encoding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_embd + pos_embd  # (B, T, C)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.linear(x)  # (B, T, vocab_size), this is the prediction for the next token in the sequence
        
        # compute the loss -> this is to eval the quality of the predictions from the model
        if targets is None:
            loss = None
        else: 
            B, T, C = logits.shape
            # Update dimensions of logits and targets to fit the loss function
            logits = logits.view(B * T, C)  # (B, T, C) -> (B * T, C)
            targets = targets.view(B * T)  # (B, T) -> (B * T)
            loss = F.cross_entropy(logits, targets)  # -> compares the predicted logits with the actual targets
        
        return logits, loss

    def generate(self, idx, max_gens=100):
        # idx is (B, T) 
        for _ in range(max_gens):
            idx_cond = idx[:, -block_size:]  # only last block_size tokens are used
            # get predictions
            logits, loss = self(idx_cond)
            # focus only on the last token/time step
            logits = logits[:, -1, :]  # (B, T, vocab_size) -> (B, vocab_size)
            # apply softmax for probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        return idx

xb, yb = get_batch('train')

m = ChadGPT(v_size) # Character model
# m = ChadGPT(v_size_words) # Word model
model2 = m.to(device)
logits2, loss2 = model2(xb, yb)

m = BigramLanguageModel(v_size) # Character model
# m = BigramLanguageModel(v_size_words) # Word model
model = m.to(device)
logits, loss = model(xb, yb)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for steps in range(max_iters):
    # Log training loss
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step: {steps}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    logits_2, loss_2 = model2(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
# print(loss.item())
print(f"Step: {steps}, Train Loss: {losses['train']:.4f}, Val Loss: {losses['val']:.4f}")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_gens=500)[0].tolist()))


# HAVE NOT ADDED GUI YET
