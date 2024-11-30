from torch import nn

from blocks import (
    ResidualConnection,
    LayerNormalization,
)



class DecoderBlock(nn.Module):

    def __init__(
        self,
        self_attention,
        cross_attention,
        feed_forward,
        dropout
    ):
        super().__init__()
        self.self_attention_block = self_attention
        self.cross_attention_block = cross_attention
        self.feed_forward_block = feed_forward
        
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x, encoder_output, enc_mask, dec_mask):
        
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, dec_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, enc_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
        
        
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, encoder_output, enc_mask, dec_mask):
        
        for layer in self.layers:
            x = layer(x, encoder_output, enc_mask, dec_mask)
        
        return self.norm(x)