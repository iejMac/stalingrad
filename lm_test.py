import math
import numpy as np
import tiktoken
import torch

from matplotlib import pyplot as plt

from stalingrad.tensor import Tensor
from stalingrad import nn
from stalingrad import optim
from stalingrad.data import fetch_shakespeare, get_batch

# Define model:
class MultiHeadAttention(nn.Module):
  def __init__(self, embed_dim, num_heads, context_length, bias=False):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.context_length = context_length
    assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

    self.w_attn = nn.Linear(embed_dim, embed_dim * 3, use_bias=bias)
    self.w_proj = nn.Linear(embed_dim, embed_dim, use_bias=bias)

    # Create causal attention masks
    lower_tri = 1.0 - np.tril(np.ones((1, 1, context_length, context_length)), k=0)
    self.attn_mask = Tensor(np.where(lower_tri, -np.inf, lower_tri), requires_grad=False)

  def causal_mask(self, scores):
    return scores + self.attn_mask

  def forward(self, x):
    B, T, C = x.shape

    emb_attn = self.w_attn(x)
    q = emb_attn[:, :, :self.embed_dim]
    k = emb_attn[:, :, self.embed_dim:self.embed_dim*2]
    v = emb_attn[:, :, self.embed_dim*2:self.embed_dim*3]

    q = q.reshape(shape=(B, T, self.num_heads, self.head_dim)).transpose(order=(0, 2, 1, 3))
    k = k.reshape(shape=(B, T, self.num_heads, self.head_dim)).transpose(order=(0, 2, 1, 3))
    v = v.reshape(shape=(B, T, self.num_heads, self.head_dim)).transpose(order=(0, 2, 1, 3))

    # TODO: is * (1/x) faster than /x ??? test it out, might be interesting
    scores = (q @ k.transpose(order=(0, 1, 3, 2))) / math.sqrt(k.shape[-1])
    scores = self.causal_mask(scores)
    scores = scores.softmax(dist_axes=(3,))
    # TODO: attn dropout
    y = scores @ v

    y = y.transpose(order=(0, 2, 1, 3)).reshape(shape=(B, T, C))
    # TODO: residual dropout
    y = self.w_proj(y)
    return y


class LayerNorm(nn.Module):
  # https://arxiv.org/abs/1607.06450
  def __init__(self, shape, epsilon=1e-5, bias=False):
    super().__init__()
    self.shape = shape
    self.epsilon = epsilon

    self.gain = Tensor(np.ones(shape))
    self.bias = Tensor(np.zeros(shape)) if bias else None

  def forward(self, x):
    # dims = tuple(range(-len(self.shape), 0))
    dims = tuple([i for i in range(len(self.shape)) if self.shape[i] != 1])

    # TODO: maybe add keepdims
    normalized_shape = x.shape[:-len(dims)] + (1,) * len(dims)

    mean = x.mean(axis=dims).reshape(shape=normalized_shape)
    var = (x**2).mean(axis=dims).reshape(shape=normalized_shape) - (mean**2)

    x_norm = self.gain * ((x - mean) / ((var + self.epsilon)**0.5))
    if self.bias is not None:
      x_norm += self.bias

    return x_norm

def gelu(x):
  # https://arxiv.org/abs/1606.08415
  return 0.5 * x * (1.0 + (math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x**3))).tanh())

class MLP(nn.Module):
  def __init__(self, embed_dim, mlp_ratio=4, bias=False):
    super().__init__()
    # TODO: needs dropout maybe??
    self.c_fc = nn.Linear(embed_dim, embed_dim * mlp_ratio, use_bias=bias)
    self.c_proj = nn.Linear(embed_dim * mlp_ratio, embed_dim, use_bias=bias)

  def forward(self, x):
    x = self.c_fc(x)
    x = gelu(x)
    x = self.c_proj(x)
    return x

class TransformerBlock(nn.Module):
  def __init__(self, embed_dim, num_heads, context_length, bias=False, mlp_ratio=4):
    super().__init__()
    self.ln_1 = LayerNorm((1, 1, embed_dim), bias=bias) # 1, 1, to adjust for not working unbroadcast
    self.attn = MultiHeadAttention(embed_dim, num_heads, context_length, bias)
    self.ln_2 = LayerNorm((1, 1, embed_dim), bias=bias)
    self.mlp = MLP(embed_dim, mlp_ratio, bias)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x


# TODO: implement
class Transformer(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return x



train_data, val_data = fetch_shakespeare(data_dir="data/shakespeare")

# training stuff
steps = 10
batch_size = 4
# TODO: think about rounding up to 50304 like karpathy https://github.com/karpathy/nanoGPT/blob/a82b33b525ca9855d705656387698e13eb8e8d4b/train.py#L144
vocab_size = 50257 

# model init and params
context_length = 7
embed_dim = 32 # 768
num_heads = 4
mlp_ratio = 1
bias = False

model = TransformerBlock(
  embed_dim=embed_dim,
  num_heads=num_heads,
  context_length=context_length,
  bias=bias,
  mlp_ratio=mlp_ratio,
)

# TODO: for weight sharing I'm trying init lm_head then tok_emb = lm_head.weight
lm_head = nn.Linear(embed_dim, vocab_size, use_bias=False)
# tok_emb = Tensor(np.random.normal(size=(vocab_size, embed_dim))) # token embeddings
tok_emb = lm_head.weight.transpose()

# TODO: look into rotary positional embeddings
# TODO: unbroadcasting is a bit scuffed, if you rmeove the 1 it complains
pos_emb = Tensor(np.random.normal(size=(1, context_length, embed_dim))) # learned positional embedding

model_params = model.parameters()
model_params["lm_head.weight"] = lm_head.weight
model_params["pos_emb"] = pos_emb

loss_func = nn.NLL(reduction="mean")
opt = optim.Adam(model_params, learning_rate=1e-2)


# TODO: make init better because things might need to be initialized differently
# as per the GPT-2 paper
# - residual projections
# - maybe token embs???

losses = []

for step in range(steps):
  bx, by = get_batch(train_data, context_length, batch_size)

  # TODO: just make CrossEntropyLoss, this is getting annoying
  by_oh = np.eye(vocab_size)[by]

  tbx, tby = Tensor(bx), Tensor(by_oh)

  # TODO: maybe we should make Slice function allow passing
  # in Tensor as inds...
  embs = tok_emb[tbx.data]
  embs += pos_emb

  out = model(embs)
  logits = lm_head(out)

  sm = logits.softmax()
  sm.backward()
  loss = loss_func(sm, tby)

  # Logging
  print(f"Step {step}: {loss.data.item()}")
  losses.append(loss.data.item())

  loss.backward()
  opt.step()
  opt.zero_grad()


# Plot the loss values
plt.plot(losses)

# Add labels and title to the plot
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Show the plot
plt.show()
