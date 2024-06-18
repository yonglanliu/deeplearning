import torch
import torch.nn as nn
from transformer.layer.Attention import MultiheadSelfAttention
from transformer.layer.PositionEncoding import PositionalEncoding
from transformer.layer.PositionwiseFeedForward import PositionwiseFeedForward
import math

class EncoderBlock(nn.Module):
    def __init__(self, num_hiddens: int, ffn_hiddens: int, num_heads: int, drop_prob, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if drop_prob == False:
            drop_prob = 0.0
        self._Attention = MultiheadSelfAttention(num_hiddens, num_heads, drop_prob)
        self._LayerNorm1 = nn.LayerNorm(normalized_shape=[num_hiddens], eps=1e-5)
        self._LayerNorm2 = nn.LayerNorm(normalized_shape=[num_hiddens], eps=1e-5)
        self._dropout1 = nn.Dropout(p=drop_prob)
        self._dropout2 = nn.Dropout(p=drop_prob)
        self._ffn = PositionwiseFeedForward(ffn_hiddens, num_hiddens, drop_prob)

    def forward(self, x, valid_lens):
        # assign residual 1
        residual_x = x

        # Attention
        x = self._Attention(x, valid_lens)
        # Dropout 1 
        x = self._dropout1(x)
        # Layer normalization 1
        x = self._LayerNorm1(x + residual_x)

        # assign residual 2
        residual_x = x

        # Feed Forward
        x = self._ffn(x)
        # Dropout 2 
        x = self._dropout2(x)
        # Add residual and layer normalization 2 
        x = self._LayerNorm2(x + residual_x)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 max_len, 
                 num_hiddens, 
                 ffn_hiddens, 
                 num_heads=8, 
                 drop_prob=0.5, 
                 num_layers=6, 
                 *args, 
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._num_hiddens = num_hiddens
        self._embedding = nn.Embedding(vocab_size, num_hiddens)
        self._pos_encoding = PositionalEncoding(num_hiddens, max_len)
        self._encoderLayers = nn.Sequential(*[EncoderBlock(num_hiddens, ffn_hiddens, num_heads, drop_prob)
                                             for _ in range(num_layers)])
        
    def forward(self, x, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        # ref: https://d2l.ai/chapter_attention-mechanisms-and-transformers/transformer.html
        x = self._embedding(x) * math.sqrt(self._num_hiddens)

        x = x + self._pos_encoding()

        for _encoderLayer in self._encoderLayers:
            x = _encoderLayer(x, valid_lens)
        return x  # (batch_size, max_len, num_hiddens)

if __name__ == "__main__":
    encoder = Encoder(200, 100, 24, 48, 8, 0.5, 2)
    x = torch.ones((2, 100), dtype=torch.long) 
    print(x.shape)
    valid_lens = torch.tensor([3, 2])
    print(valid_lens.shape)
    out = encoder(x, valid_lens)
    #print(out)
    print(out.shape)