import math
from torch import nn


class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(
        self,
        d_model: int,
        h: int,
        dropout: float
    ):
        
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        
        assert d_model % h == 0, "d_model is not divisible by h"
        
        self.d_k = d_model / h
        
        # Query, Key, Value
        self.w_q = nn.Linear(d_model, d_model) 
        self.w_k = nn.Linear(d_model, d_model) 
        self.w_v = nn.Linear(d_model, d_model) 
        
        # Final Output Weights
        self.w_o = nn.Linear(d_model, d_model)
        
        
    @staticmethod
    def attention(query, key, value, mask, dropout):
        
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask:
            attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores 

    
    def forward(self, q, k, v, mask):
        
        query = self.w_q(q) # q' (seq_len, d_model) = q (seq_len, d_model) * w_q (d_model, d_model)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # (batch, seq_len, d_model) =>(batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
         
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Concat
        # (batch, h, seq_len, d_k) => (batch, seq_len, h, d_k) => (batch, seq_len, d_model)
        x = x.transpose(1,2)
        x = x.contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
                
        return (x @ self.w_o), self.attention_scores
        
        
        
        
        
        
        