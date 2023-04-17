import math
import numpy as np
import tiktoken
import torch

from dataclasses import dataclass
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


@dataclass
class GPTConfig:
  context_length: int = 7
  vocab_size: int = 50257
  num_layers: int = 2
  num_heads: int = 4
  embed_dim: int = 32
  mlp_ratio: int = 4
  bias: bool = False


class GPT(nn.Module):
  def __init__(self, config):
    super().__init__()
    # TODO: add dropout
    # TODO: for weight sharing I'm trying init lm_head then tok_emb = lm_head.weight
    # TODO: check if this even works
    self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, use_bias=config.bias)
    # tok_emb = Tensor(np.random.normal(size=(vocab_size, embed_dim))) # token embeddings
    self.tok_emb = self.lm_head.weight.transpose()

    # TODO: look into rotary positional embeddings
    # TODO: unbroadcasting is a bit scuffed, if you rmeove the 1 it complains
    self.pos_emb = Tensor(np.random.normal(size=(1, config.context_length, config.embed_dim))) # learned positional embedding

    self.blocks = []
    for i in range(config.num_layers):
      block = TransformerBlock(
        embed_dim = config.embed_dim,
        num_heads = config.num_heads,
        context_length = config.context_length,
        bias = config.bias,
        mlp_ratio = config.mlp_ratio,
      )
      # TODO: need to make something like ModuleList
      setattr(self, f"TransformerBlock_{i}", block) # so optimizer can get params
      self.blocks.append(block)

    ln_f = LayerNorm((1, 1, config.embed_dim), bias=config.bias)

  def forward(self, x):
    # TODO: maybe we should make Slice function allow passing
    # in Tensor as inds...
    x = self.tok_emb[x.data]
    x += self.pos_emb

    for block in self.blocks:
      x = block(x)

    x = self.lm_head(x)
    return x


train_data, val_data = fetch_shakespeare(data_dir="data/shakespeare", tokenizer="char")

# training stuff
steps = 10
batch_size = 8

# TODO: think about rounding up vocab size to 50304 like karpathy https://github.com/karpathy/nanoGPT/blob/a82b33b525ca9855d705656387698e13eb8e8d4b/train.py#L144
model_args = {
  "context_length": 4,
  "vocab_size": 65,  # 50257,
  "num_layers": 2,
  "num_heads": 4,
  "embed_dim": 64,
  "mlp_ratio": 2,
  "bias": False,
}

config = GPTConfig(**model_args)
model = GPT(config)

loss_func = nn.NLL(reduction="mean")
opt = optim.Adam(model.parameters(), learning_rate=1e-2)

# TODO: make init better because things might need to be initialized differently
# as per the GPT-2 paper
# - residual projections
# - maybe token embs???

losses = []

for step in range(steps):
  bx, by = get_batch(train_data, config.context_length, batch_size)
  # TODO: just make CrossEntropyLoss, this is getting annoying
  by_oh = np.eye(config.vocab_size)[by]
  tbx, tby = Tensor(bx), Tensor(by_oh)

  logits = model(tbx)
  sm = logits.softmax()

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
