import torch
import torch.nn as nn
from transformer.layer.Attention import MultiheadSelfAttention, MultiheadCrossAttention
from transformer.layer.PositionEncoding import PositionalEncoding
from transformer.layer.PositionwiseFeedForward import PositionwiseFeedForward
import math

class DecoderBlock(nn.Module):
    def __init__(self, num_hiddens: int, ffn_hiddens: int, num_heads: int, drop_prob, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if drop_prob == False:
            drop_prob = 0.0
        self._SelfAttention = MultiheadSelfAttention(num_hiddens, num_heads, drop_prob)
        self._LayerNorm1 = nn.LayerNorm(normalized_shape=[num_hiddens], eps=1e-5)
        self._LayerNorm2 = nn.LayerNorm(normalized_shape=[num_hiddens], eps=1e-5)
        self._LayerNorm3 = nn.LayerNorm(normalized_shape=[num_hiddens], eps=1e-5)
        self._dropout1 = nn.Dropout(p=drop_prob)
        self._dropout2 = nn.Dropout(p=drop_prob)
        self._dropout3 = nn.Dropout(p=drop_prob)
        self._ffn = PositionwiseFeedForward(ffn_hiddens, num_hiddens, drop_prob)

        self._CrossAttention = MultiheadCrossAttention(num_hiddens, num_heads, drop_prob)

    def forward(self, x, x_valid_lens, y):
        if self.training:
            batch_size, y_num_steps, _ = y.shape
            y_valid_lens = torch.arange(1, y_num_steps + 1, device=y.device).repeat(batch_size, 1)
        else:
            y_valid_lens = None

        residual_y = y   
        y = self._SelfAttention(y, y_valid_lens)  
        y = self._dropout1(y)    
        y = self._LayerNorm1(y + residual_y)  
        residual_y = y   
        y = self._CrossAttention(x, y, x_valid_lens)   
        y = self._dropout2(y)   
        y = self._LayerNorm2(y + residual_y)   
        residual_y = y      
        y = self._ffn(y)      
        y = self._dropout3(y)    
        y = self._LayerNorm3(y + residual_y)    
        return y

class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 max_len, 
                 num_hiddens, 
                 ffn_hiddens,
                 num_heads=8,
                 drop_prob=0.5,
                 num_layers=8,
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._num_hiddens = num_heads
        self._embedding = nn.Embedding(vocab_size, num_hiddens)
        self._pos_encoding = PositionalEncoding(num_hiddens, max_len)
        self._decoderLayers = nn.Sequential(*[DecoderBlock(num_hiddens, ffn_hiddens, num_heads, drop_prob)
                                             for _ in range(num_layers)])
        
    def forward(self, x, x_valid_lens, y):
        y = self._embedding(y) * math.sqrt(self._num_hiddens)
        y = y + self._pos_encoding()
        #print(y.shape)
        for _decoderLayer in self._decoderLayers:
            y = _decoderLayer(x, x_valid_lens, y)
        return y


if __name__ == "__main__":

    num_hiddens= 512
    num_heads = 8
    drop_prob = 0.5
    batch_size = 30
    max_len = 100
    ffn_hiddens = 2048
    num_layers = 5
    x_valid_lens = torch.arange(5, 35)
    vocab_size=200

    x = torch.randn( (batch_size, max_len, num_hiddens)) 
    y = torch.ones( (batch_size, max_len), dtype=torch.long ) 
    decoder = Decoder(vocab_size, max_len, num_hiddens, ffn_hiddens, num_heads, drop_prob, num_layers)
    out = decoder(x, x_valid_lens, y)
    print(out.shape)