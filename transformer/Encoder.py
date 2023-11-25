import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import Attentions

class PositionwiseFFN(nn.Module):
	def __init__(self, d_model, hidden, drop_prob=0.1):
		super(PositionwiseFFN, self):
		self.linear_1 = nn.Linear(d_model, hidden)
		self.linear_2 = nn.Linear(hidden, d_model)
		self.relu = nn.ReLu()
		self.dropout = nn.Dropout(p=drop_prob)

	def forward(self, x):
		x = self.linear_1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.linear_2(x)
		return x

class LayerNorm():
	def __init__(self, parameters_shape, eps=1e-5):
		self.parameters_shape = parameters_shape
		self.eps = eps
		self.gamma = nn.Parameter(torch.ones(parameters_shape))
		self.beta = nn.Parameter(torch.zeros(parameters_shape))

	def forward(self, input)
		dims = [-(i+1) for i in range(len(parameters_shape))]
		mean = torch.mean(input, dim=dims, keepdim=True)
		var = torch.mean((input - mean)**2, dim=dims, keepdim=True)
		std = torch.sqrt(var + self.eps)
		y = (input - mean) / std
		out = self.gamma * y + self.beta
		return out

class EncoderLayer(nn.Module):
	def __init__(self, d_model, num_heads, fnn_hidden, mask=False, drop_prob=0.1):
		super(EncoderLayer, self).__init__
		self.attention = Attentions.MultiheadSelfAttention(d_model, num_heads, mask=False)
		self.ffn = PositionwiseFFN(d_model, ffn_hidden, drop_prob=0.1)
		self.norm1 = LayerNorm(parameters_shape=[d_model])
		self.norm2 = LayerNorm(parameters_shape=[d_model])
		self.dropout1 = nn.Dropout(drop_prob)
		self.dropout2 = nn.Dropout(drop_prob)

	def forward(self, x):
		x_residue = x
		x = self.attention(x)
		x = self.dropout1(x)
		x = self.norm1(x_residue + x)

		x_residue = x
		x = self.ffn(x)
		x = self.dropout2(x)
		x = self.norm2(x_residue + x)

		return x

class Encoder(nn.Moduel):
	def __init__(self, d_model, num_heads, fnn_hidden, num_layers, mask=False, drop_prob=0.1):
		super(Encoder, self).__init__
		self.layers = nn.Sequential(*[EncoderLayer(d_model, num_heads, fnn_hidden, mask, drop_prob) 
			for _ in range(num_layers)])

	def forward(self, x):
		x = self.layers(x)
		return x