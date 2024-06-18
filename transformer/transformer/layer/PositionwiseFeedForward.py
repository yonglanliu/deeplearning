import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, ffn_num_hiddens, num_hiddens, drop_prob=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.linear2 = nn.Linear(ffn_num_hiddens, num_hiddens)  # d_model
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
