import torch
from torch import nn

from blocks import *
from attention import MultiHeadAttentionBlock
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock


class Transformer(nn.Module):
    
    def __init__(
        self,
        encoder,
        decoder,
        encoder_embeddings,
        decoder_embeddings,
        encoder_pe,
        decoder_pe,
        projection_layer
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_embeddings = encoder_embeddings
        self.decoder_embeddings = decoder_embeddings
        self.encoder_pe = encoder_pe
        self.decoder_pe = decoder_pe
        self.projection_layer = projection_layer
        
        
        
    def encode(self, enc, enc_mask):
        enc = self.encoder_embeddings(enc)
        enc = self.encoder_pe(enc)
        return self.encoder(enc, enc_mask)
    
    def decode(self, encoder_output, enc_mask, dec, dec_mask):
        dec = self.decoder_embeddings(dec)
        dec = self.decoder_pe(dec)
        return self.decoder(dec, encoder_output, enc_mask, dec_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
    
    
def build_transformer(
    src_vocab_size,
    tgt_vocab_size,
    src_seq_len,
    tgt_seq_len,
    d_model: int  = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048 
):
    
    # Embedding Layers
    encoder_embeddings = InputEmbeddings(d_model, src_vocab_size)
    decoder_embeddings = InputEmbeddings(d_model, tgt_vocab_size)
    
    # Positional Encodings
    encoder_pe = PositionalEncoding(d_model, src_seq_len, dropout)
    decoder_pe = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # Encoder Blocks
    encoder_blocks = []
    for _ in range(N):
        
        encoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBLock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
        
    # Decoder Blocks
    decoder_blocks = []
    for _ in range(N):
        
        decoder_self_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBLock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention, decoder_cross_attention, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
        
        
    # Create Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    
    # Projection Layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Transformer
    transfomer = Transformer(
        encoder,
        decoder,
        encoder_embeddings,
        decoder_embeddings,
        encoder_pe,
        decoder_pe,
        projection_layer
    )
    
    # Initialize the parameters
    
    for p in transfomer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transfomer
    
        