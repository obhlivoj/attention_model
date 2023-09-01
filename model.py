import torch
import torch.nn as nn
import numpy as np
from typing import Union, Callable

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, input_dim: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.embedding = nn.Linear(input_dim, d_model)
        # layer normalization or scaling?

    # x -- (batch, seq_len, num_features) -> (batch, seq_len, d_model)
    def forward(self, x):
        return self.embedding(x) # * np.sqrt(self.d_model) -- for language models
        # droupout

# class OutputEmbedding(nn.Module):

#     def __init__(self, d_model: int, out_seq: int, seq_len: int) -> None:
#         super().__init__()
#         self.d_model = d_model
#         self.out_seq = out_seq
#         self.seq_len = seq_len
#         self.embedding = nn.Linear(out_seq, d_model * seq_len)
#         # layer normalization or scaling?

#      # x -- (batch, out_seq) -> (batch, seq_len * d_model) -> (batch, seq_len, d_model)
#     def forward(self, x):
#         x = self.embedding(x) # * np.sqrt(self.d_model) -- for language models
#         x = x.view(x.shape[0], self.seq_len, self.d_model)
#         return x 
#         # droupout
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, droput: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(droput)

        # create a matrix of positional encodings -- (seq_len, d_model)
        pos_enc = torch.zeros(seq_len, d_model)
        # create a vector -- (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)) ## stability reasons
        # apply sine and cosine
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        # add dimension
        pos_enc = pos_enc.unsqueeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pe', pos_enc)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1)) # multiplicative param "gain"
        self.bias = nn.Parameter(torch.zeros(1)) # additive param "bias"

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.gain + self.bias
    
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor, activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu) -> torch.Tensor:
        # [batch, seq_len, d_model] ->  [batch, seq_len, d_ff] -> [batch, seq_len, d_model]
        return self.linear_2(self.dropout(activation(self.linear_1(x)))) 
    
class MutliHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, nheads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.nheads = nheads
        assert d_model % nheads == 0, "d_model must be divisible by nheads"

        self.d_k = d_model // nheads
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # [batch, nheads, seq_len, d_k] - > [batch, nheads, seq_len, seq_len]
        attention_scores = (query @ key.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # [batch, nheads, seq_len, seq_len] - > [batch, nheads, seq_len, d_k]
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        query = self.w_q(q) ## [batch, seq_len, d_model] ->  [batch, seq_len, d_model]
        key = self.w_k(k)
        value = self.w_v(v)

        # [batch, seq_len, d_model] -> [batch, seq_len, nheads, d_k] -> [batch, nheads, seq_len, d_k]
        query = query.view(query.shape[0], query.shape[1], self.nheads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.nheads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.nheads, self.d_k).transpose(1, 2)

        x, self.attention_scores = MutliHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # [batch, nheads, seq_len, d_k] -> [batch, seq_len, nheads, d_k] -> [batch, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.nheads * self.d_k)

        # [batch, seq_len, d_model] -> [batch, seq_len, d_model]
        return self.w_o(x)
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) ## first normalize, then use the sublayer
    
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MutliHeadAttentionBlock, ff_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.ff_block = ff_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    # source mask - hide interactions with padding
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.ff_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MutliHeadAttentionBlock, cross_attention_block: MutliHeadAttentionBlock, ff_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.ff_block = ff_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_out, encoder_out, None))
        x = self.residual_connections[2](x, self.ff_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)
        return self.norm(x)
    
class LinearProjection(nn.Module):

    def __init__(self, d_model: int, n_tgt: int, seq_len: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_tgt = n_tgt
        self.seq_len = seq_len
        self.lin_project = nn.Linear(d_model, n_tgt)

    def forward(self, x):
        # [batch, seq_len, d_model] -> [batch, seq_len, n_tgt]
        return self.lin_project(x)
        
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, lin_proj: LinearProjection) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = lin_proj

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_out, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)
    

def build_transformer(num_features: int, n_tgt: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, Nx: int = 6, nheads: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # create embedding layers
    src_embed = InputEmbedding(d_model, num_features)
    tgt_embed = InputEmbedding(d_model, n_tgt)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # create the encoder blocks
    encoder_blocks = []
    for _ in range(Nx):
        encoder_self_attention_block = MutliHeadAttentionBlock(d_model, nheads, dropout)
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, ff_block, dropout)
        encoder_blocks.append(encoder_block)
    
    # create the decoder blocks
    decoder_blocks = []
    for _ in range(Nx):
        decoder_self_attention_block = MutliHeadAttentionBlock(d_model, nheads, dropout)
        decoder_cross_attention_block = MutliHeadAttentionBlock(d_model, nheads, dropout)
        ff_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, ff_block, dropout)
        decoder_blocks.append(decoder_block)

    # create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # create the projection layer
    projection_layer = LinearProjection(d_model, n_tgt, tgt_seq_len)

    # create the transformer
    transformer_model = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # weights initialization (Xavier - uniform)
    for p in transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer_model