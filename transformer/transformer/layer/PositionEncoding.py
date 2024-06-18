import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len=1000, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_hiddens = num_hiddens
        self.max_len = max_len
    
    def forward(self):
        even_i = torch.arange(0, self.num_hiddens, 2).float()
        denominator = torch.pow(10000, even_i/self.num_hiddens)
        position = torch.arange(self.max_len).reshape(self.max_len, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE
    
if __name__ == "__main__":
    pe = PositionalEncoding(num_hiddens=6, max_len=10)
    print(pe.forward())