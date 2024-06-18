import torch
import torch.nn as nn
import torch.nn.functional as F
import math

#--------------------------------------
#     Scaled Dot Product Attention      
#--------------------------------------
class DotProductAttention(nn.Module):
    """
    Dot Product Attention Scoring Function
    Attention is all you need: https://arxiv.org/abs/1706.03762
    """
    def __init__(self, drop_prob, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(drop_prob)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, q, k, v, valid_lens=None):
        """
        Shape of queries (q): (batch_size * num_heads, no. of queries, d)
        Shape of keys (k): (batch_size * num_heads, no. of key-value pairs, d)
        Shape of values (v): (batch_size * num_heads, no. of key-value pairs, value dimension)
        Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
        """
        d = k.size()[-1]  # dimension of k.  q and k have the same dimension
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d) # (batch_size * num_heads, max_len, max_len)
        scores = scores.to(self.device)
        self.attention_weights = self._masked_softmax(scores, valid_lens) # (batch_size * num_heads, max_len, max_len)
        #print(f"\"scores\" shape is {scores.shape}")
        return torch.matmul(self.dropout(self.attention_weights), v)
    
    def _masked_softmax(self, X, valid_lens): 
        """Perform softmax operation by masking elements on the last axis.
        X: (batch_size * num_heads, max_len, max_len)
        valid_lens: batch_size * num_heads 
        """
        if valid_lens is None:
            return F.softmax(X, dim=-1)
        else:
            shape = X.shape 
            if valid_lens.dim() == 1:
                valid_lens = torch.repeat_interleave(valid_lens, shape[1]) 
            else:
                valid_lens = valid_lens.reshape(-1)
            # print(f"\"Valid_len\" shape is {valid_lens.shape}")  # batch_size * num_heads * max_len
            # On the last axis, replace masked elements with a very large negative
            # value, whose exponentiation outputs 0
            X = self._sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
            return F.softmax(X.reshape(shape), dim=-1)

    def _sequence_mask(self, X, valid_lens, value=0):
        """
        X: (batch_size * num_heads, max_len)
        valid_lens: batch_size * num_heads * max_len
        mask: (batch_size * num_heads, max_len)
        """
        maxlen = X.size(1) # obtain max_len
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[None, :] < valid_lens[:, None]
        X[~mask] = value
        return X
    
#--------------------------------------
#       Multi-Head Self-Attention             
#--------------------------------------
class MultiheadSelfAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, drop_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.head_dim = num_hiddens // num_heads
        self.qkv_layer = nn.Linear(num_hiddens, 3 * num_hiddens) 
        self.linear_layer = nn.Linear(num_hiddens, num_hiddens)
        self.attention = DotProductAttention(drop_prob=drop_prob)

    def forward(self, x, val_lens):
        """
        val_lens: batch_size
        """
        batch_size, max_len, num_hiddens = x.size()

		# obtain q, k, v
        qkv = self.qkv_layer(x)
        qkv = torch.reshape(qkv, shape=(batch_size, max_len, self.num_heads, 3 * self.head_dim))
        qkv = qkv.permute(0, 2, 1, 3)

        # split last dimension into q, k, v
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(-1, q.shape[2], q.shape[3])
        k = k.reshape(-1, k.shape[2], k.shape[3])
        v = v.reshape(-1, v.shape[2], v.shape[3])

        if val_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            val_lens = torch.repeat_interleave(val_lens, repeats=self.num_heads, dim=0)  # batch_size * num_heads

        values = self.attention(q, k, v, val_lens)
        #print(f"\"Values\" shape is {values.shape}")

		# Flatten all heads together: concate
        values = values.reshape(-1, self.num_heads, values.shape[1], values.shape[2])  # (batch_size, num_heads, max_len, head_dim)
        #print(f"\"Values after reshape\" shape is {values.shape}")
        values = values.permute(0, 2, 1, 3) # (batch_size, max_len, num_heads, head_dim)
        #print(f"\"Values after reshape\" shape is {values.shape}")
        values = values.reshape(values.shape[0], values.shape[1], -1)  # (batch_size, max_len, num_hiddens)
        #print(f"\"Values after reshape\" shape is {values.shape}")

		# Feed into a linear layer to restore the dimension
        out = self.linear_layer(values)
        return out
    
# --------------------------------
#     Multi-Head Cross Attention
# --------------------------------
class MultiheadCrossAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, drop_prob, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.head_dim = num_hiddens // num_heads
        self.kv_layer = nn.Linear(num_hiddens, 2 * num_hiddens) 
        self.q_layer = nn.Linear(num_hiddens, num_hiddens) 
        self.linear_layer = nn.Linear(num_hiddens, num_hiddens)
        self.attention = DotProductAttention(drop_prob=drop_prob)

    def forward(self, x, y, val_lens):
        """
        val_lens: batch_size
        """
        batch_size, max_len, num_hiddens = x.size()

		# obtain q, k, v
        kv = self.kv_layer(x)
        kv = torch.reshape(kv, shape=(batch_size, max_len, self.num_heads, 2 * self.head_dim))
        kv = kv.permute(0, 2, 1, 3)

        # split last dimension into q, k, v
        k, v = kv.chunk(2, dim=-1)
        k = k.reshape(-1, k.shape[2], k.shape[3])
        v = v.reshape(-1, v.shape[2], v.shape[3])

        q = self.q_layer(y)
        q = torch.reshape(q, shape=(batch_size, max_len, self.num_heads, self.head_dim))
        q = q.permute(0, 2, 1, 3)
        q = q.reshape(-1, q.shape[2], q.shape[3])

        if val_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            val_lens = torch.repeat_interleave(val_lens, repeats=self.num_heads, dim=0)  # batch_size * num_heads

        values = self.attention(q, k, v, val_lens)
        #print(f"\"Values\" shape is {values.shape}")

		# Flatten all heads together: concate
        values = values.reshape(-1, self.num_heads, values.shape[1], values.shape[2])  # (batch_size, num_heads, max_len, head_dim)
        #print(f"\"Values after reshape\" shape is {values.shape}")
        values = values.permute(0, 2, 1, 3) # (batch_size, max_len, num_heads, head_dim)
        #print(f"\"Values after reshape\" shape is {values.shape}")
        values = values.reshape(values.shape[0], values.shape[1], -1)  # (batch_size, max_len, num_hiddens)
        #print(f"\"Values after reshape\" shape is {values.shape}")

		# Feed into a linear layer to restore the dimension
        out = self.linear_layer(values)
        return out

if __name__ == "__main__":
    # Dot Self-Attention
    """    
    queries = torch.normal(0, 1, (2, 1, 2))
    keys = torch.normal(0, 1, (2, 10, 2))
    values = torch.normal(0, 1, (2, 10, 4))
    valid_lens = torch.tensor([2, 6])

    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    """
    # Multi-Head Attention
    num_hiddens = 512
    num_heads = 8
    batch_size = 30
    max_len = 5
    drop_prob = 0.2
    val_len = torch.arange(0, 30)
    x = torch.randn((batch_size, max_len, num_hiddens))
    mulAtten = MultiheadSelfAttention(num_hiddens, num_heads, drop_prob)
    out = mulAtten(x, val_len)
    #print(out)
    print(out.shape)