import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import Attentions as at


d_model = 512
num_heads = 8
drop_prob = 0.1
batch_size = 30
max_sequence_length = 200
ffn_hidden = 2048
num_layers = 5

x = torch.randn((batch_size, max_sequence_length, d_model)) # English sentence positional encoded
y = torch.randn((batch_size, max_sequence_length, d_model)) # 

class SequentialDecoder(nn.Sequential):
	def forward(self, *inputs):
		x, y, mask = inputs
		for module in self._modules.values():
			y = module(x, y, mask)
		return y

class Decoder(nn.Module):
	def __init__(self, d_model, ffn_hidden, num_heads, drop_prob, num_layers):
		super(Decoder, self).__init__
		self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)])