import torch
import torch.nn as nn
from torch.nn import functional as F

# hypers

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200


torch.manual_seed(40)


with open("input.txt", "r") as f:
    text = f.read()


chars = sorted(list(set(text)))
vocab_size = len(chars)
encodings = {ch: i for i, ch in enumerate(chars)}
decodings = {i: ch for i, ch in enumerate(chars)}
encode = lambda enc_str: [encodings[char] for char in enc_str]
decode = lambda dec_int: ''.join([decodings[i] for i in dec_int])

data = torch.tensor(encode(text), dtype=torch.long)
# train-val split
split = int(0.9*len(data))
train_data = data[:split]
val_data = data[split:]


def get_batch(split):
    data = train_data if split=='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:block_size+i] for i in ix])
    y = torch.stack([data[i+1:block_size+i+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out



class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding(idx) # (B,T,C) (Batch, Time, Channel)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens=4):
        
        for _ in range(max_new_tokens):
            
            logits, loss = self(idx)
            # focus on the last time step
            logits = logits[:, -1, :] # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)


for iter in range(max_iters):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    xb, yb = get_batch('train')
    
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
idx = torch.zeros((1,1), dtype=torch.long)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))






