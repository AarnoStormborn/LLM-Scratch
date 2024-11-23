import torch
import math
from torch import nn

class InputEmbeddings(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # As mentioned in the paper
        
        
class PositionalEncoding(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        seq_len: int,
        dropout: float
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Matrix of shape (seq_len, d_model)
        pe = torch.zeros(self.seq_len, self.d_model)
        
        # Formula of Positional Encoding: pos * exp(-(log(10000) / d_model))
        
        # First Term of formula
        # Vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtpye=torch.float).unsqueeze(1)
        
        # Second Term of formula
        # Inner expression of the formula
        inner_exp = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even positions and cos to odd positions
        pe[:, 0::2] = torch.sin(position * inner_exp)
        pe[:, 1::2] = torch.cos(position * inner_exp)
        
        pe = pe.unsqueeze(0) # (seq_len, d_model) => (1, seq_len, d_model)
        
        self.register_buffer('pe', pe) # Save the tensor in the Model
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)
        return self.dropout(x)
    
    
    
class LayerNormalization(nn.Module):
    
    def __init__(
        self,
        eps: float = 1e-6
    ):
        super().__init__()
        self.eps = eps
        
        # Learnable parameters
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # Additive
        
    def forward(self, x):
        
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        return (self.alpha * ((x - mean)/ (std + self.eps))) + self.bias
    
    
class FeedForwardBLock(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        
        # In paper, FFN(x) = max(0, W1x + b1)W2 + b2
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
    
    
class ResidualConnection(nn.Module):
    
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
    
class ProjectionLayer(nn.Module):
    
    def __init__(
        self,
        d_model,
        vocab_size
    ):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    
    def forward(self, x):
        
        return torch.log_softmax(self.proj(x), dim=-1)
                
    
    
        
        
        
    
 