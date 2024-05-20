#gpt from the video lecture
verbose = False


with open('input.txt', 'r', encoding ='utf-8') as f:
    text = f.read()

# Make vocab
chars = sorted(list(set(text)))
vocab_size = len(chars)
assert vocab_size == 65

# Encoders, decoders

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

test_string = "Hey Marek!"
assert decode(encode(test_string)) == test_string

# google sentencepiece tokenizer 
# openai tiktoken, eg enc = tiktoken.get_encoding("gpt2")
# enc.decode(enc.encode(test_string)) == test_string

import torch


# Create training and test datasets
data = torch.tensor(encode(text), dtype=torch.long)
train_len = int(0.9*len(data))
train_data = data[:train_len]
val_data = data[train_len:]

# Create batches

block_size = 8 # context length

transformer_input = train_data[:block_size]
transformer_target = train_data[1:block_size+1]
x = transformer_input
y = transformer_target

# Nice print out of decoded characters
if verbose:
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        x_chars = decode(context.tolist())
        y_chars = decode([target.tolist()])
        print(f'When input is {x_chars} target is {y_chars}')

# Batching

torch.manual_seed(1337)
batch_size = 4
block_size = 8 # context length

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # get the random index between across the dataset
    # in fact get (batch_size, ) of them.
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Bigram language model

import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

class BigramLanguageModel(nn.Module):
    # (B,T,C) batch x time x channel
    def __init__(self, vocab_size):
        super().__init__()
        # why is vocab_size == embedding size? I guess for convenience?
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # for a multidimensional input pytorch expects:
            # minibatch , C (classes), d1, d2, d3.. dimensions
            # B T C
            # it wants C to be second in F.cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _loss = self(idx) # calling models on token indices
            logits = logits[:, -1, :] # becomes (B,C)
            probs = F.softmax(logits, dim=-1) # (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    # before I move on to the next step I need to print out those shapes and understand better
    # the context, the batch, the channel.
    # as well as get an answer to vocab_size - which is vocab_sie and which is embedding_dim

# test embeddings
if verbose:
    token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    print(token_embedding_table(torch.LongTensor([5,2])))

m = BigramLanguageModel(vocab_size)
xb, yb = get_batch('train')
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)
# for uninitialised network loss of 65 characters should be:
# -ln(1/65)

# Untrained generation:
z = torch.zeros((1,1), dtype=torch.long)
output = m.generate(z, max_new_tokens=100)[0].tolist()
print(decode(output))


