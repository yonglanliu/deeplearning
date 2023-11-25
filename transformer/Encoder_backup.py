import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def _sequence_mask(scaled):
    """
    Mask is required for decoder, but it is not required for encoder
    Mask is to ensure words that don't get context from words generated in the future
    """
    mask = torch.full(scaled.size(), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    return mask

def scaled_dot_product_attention(q, k, v, mask=False):
    """
    Attention(Q, K, V) = matmul(softmax(matmul(Q, K.T)/sqrt(d)), V)
    q, k, v = batch_size * num_head * sequence_length * head_dim
    """
    # head_dim
    d_k = q.size()[-1]  # 64
    
    # batch_size * num_head * sequence_length * sequence_length
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k) 

    if mask == True:
        # batch_size * num_head * sequence_length * sequence_length
        scaled += __sequence_mask(scaled) 

    # get the probability of how much the attention is
    attentions = F.softmax(scaled, dim=-1) 
    values = torch.matmul(attentions, v) # 30 * 8 * 200 * 64
    return values, attentions

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mask):
        super(MultiHeadAttention, self).__init__
        self.d_model = d_model # 512
        self.num_heads = num_heads # 8
        self.heads_dim = d_model // num_heads # 64
        self.qkv_layer = nn.Linear(d_model, 3 * d_model) # 512 * 1536
        self.linear_layer = nn.Linear(d_model, d_model) # 512 * 512
        self.mask = mask # False for Encoder, True for Decoder

    def forward(self, x):
        batch_size, sequence_length, d_model = x.size() # 32 * 200 * 512
        qkv = self.qkv_layer(x) # 32 * 200 * 1536
        qkv = torch.reshape(qkv, shape=(batch_size, sequence_length, self.num_heads, self.heads_dim)) # 32 * 200 * 8 * 192
        qkv = qkv.permute(0, 2, 1, 3) # 32 * 8 * 200 * 192
        q, k, v = qkv.chunk(3, dim=-1) # 32 * 8 * 200 * 64
        values, attentions = scaled_dot_product_attention(q, k, v, mask=self.mask) # 30 * 8 * 200 * 64
        values = torch.reshape(values, shape=(batch_size, sequence_length, self.heads * self.head_dim)) # 30 * 200 * 512
        out = linear_layer(values) # 30 * 200 * 512
        return out

class LayerNorm():
    def __init__(self, parameters_shape, eps=1e-5):
        self.parameters_shape = parameters_shape #[512]
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape)) # [512]
        self.beta = nn.Parameter(torch.zeros(parameters_shape)) # [512]

    def forward(self, input): 
        dims = [-(i+1) for i in range(len(self.parameters_shape))] [-1]
        mean = torch.mean(input, dim=dims, keepdim=True) # 30 * 200 *1
        var = torch.mean((input - mean)**2, dim=dims, keepdim=True) # 30 * 200 *1
        std = torch.sqrt(var + self.eps) # 30 * 200 * 1
        y = (input - mean) / std # 30 * 200 * 512
        out = self.gamma * y + self.beta # 30 * 200 * 512
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLu()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers, mask=False):
        super(EncoderLayer, self).__init__
        self.attention = MultiHeadAttention(d_model, num_heads, mask)
        self.norm1 = LayerNorm(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x):
        residue_x = x # 30 * 200 * 512
        x = self.attention(x) # 30 * 200 * 512
        x = self.dropout1(x) # 30 * 200 * 512
        x = self.norm1(x + residue_x) # 30 * 200 * 512
        residue_x = x # 30 * 200 * 512
        x = self.ffn(x)
        x = dropout2(x)
        x = norm2(x + residue_x)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
        super(Encoder, self).__init__
        self.layers = nn.Sequential(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob, num_layers) 
                                    for _ in range(num_layers)])
    def forward(self, x):
        x = self.layers(x)
        return x