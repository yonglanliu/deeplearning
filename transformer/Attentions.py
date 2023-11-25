import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Define a mask function
def _sequence_mask(scaled):
	mask_dim = scaled.size()
	mask = torch.full(scaled.mask_dim, dtype="-inf")
	mask = torch.triu(mask, diagonal=1)
	return mask

################################################################
#                    Scaled Dot Product Attention              #
################################################################
def scaled_dot_product_attention(q, k, v, mask=False):
	"""
	paper link: https://arxiv.org/abs/1706.03762v7
	"""
	d_k = k.size()[-1]
	scaled = torch.matmul(Q, k.transpose(-1, -2)) / math.sqrt(d_k)
	if mask == True:
		print(f'Mask is applied. Dimension is {_sequence_mask(scaled).size()}')
		scaled = _sequence_mask(scaled)
	attentions = F.softmax(scaled, dim=-1)
	values = torch.matmul(attentions, v)
	return values, attentions

#######################################################################
#                          Multi-Head Self-Attention                  #
#######################################################################
class MultiheadSelfAttention(nn.Module):
	def __init__(self, d_model, num_heads, mask=False):
		super(MultiheadAttention, self).__init__
		self.d_model = d_model
		self.num_heads = num_heads
		self.head_dim = d_model // num_heads
		self.qkv_layer = nn.Linear(d_model, 3 * d_model)
		self.linear_layer = nn.Linear(d_model, d_model)
		self.mask = mask

	def forward(self, x):
		batch_size, sequence_length, d_model = x.size()

		# obtain q, k, v
		qkv = self.qkv_layer(x)
		qkv = torch.reshape(qkv, shape=(batch_size, sequence_length, self.num_heads, 3 * self.head_dim))
		qkv = qkv.permute(0, 2, 1, 3)
		q, k, v = qkv.chunk(3, dim=-1)
		values, attentions = scaled_dot_product_attention(q, k, v, mask=self.mask)

		# Flatten all heads together
		values = torch.reshape(values, shape=(batch_size, sequence_length, d_model))

		# Feed into a linear layer to restore the dimension
		out = self.linear_layer(values)
		return out

########################################################################
#                         Multi-Head Cross-Attention                   #
########################################################################
class MultiheadCrossAttention(nn.Module):
	def __init__(self, d_model, num_heads):
		self.d_model = d_model
		self.num_heads = num_heads
		self.head_dim = d_model // num_heads
		self.qk_layer = nn.Linear(d_model, 2 * d_model)
		self.v_layer = nn.Linear(d_model, d_model)
		self.linear_layer = nn.Linear(d_model, d_model)

	def foward(self, x, y):

		batch_size, sequence_length, d_model = x.size()

		# obtain q, k, v
		qk = self.kv_layer(x)
		qk = torch.reshape(qk, shape=(batch_size, sequence_length, self.num_heads, 2 * self.head_dim))
		qk = qk.permute(0, 2, 1, 3)
		q, k = qk.chunk(2, dim=-1)
		v = self.v_layer(y)
		v = torch.reshape(v, shape=(batch_size, sequence_length, self.num_heads, self.head_dim))
		v = v.permute(0, 2, 1, 3)

		# apply self-attention
		values, attentions = scaled_dot_product_attention(q, k, v, mask=False)

		# Flatten all heads together
		values = torch.reshape(values, shape=(batch_size, sequence_length, d_model))

		# Feed into a linear layer to restore the dimension
		out = self.linear_layer(values)
		return out



