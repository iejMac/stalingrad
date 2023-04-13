import math
import numpy as np
import tiktoken

from stalingrad.tensor import Tensor
from stalingrad import nn
from stalingrad import optim
from stalingrad.data import fetch_shakespeare, get_batch

# Define model:
# TODO: implement
class Attention(nn.Module):
  def __init__(self, d_model, context_length):
    super().__init__()
    self.d_model = d_model
    self.context_length = context_length

    self.w_attn = nn.Linear(d_model, d_model * 3, use_bias=False)
    self.w_proj = nn.Linear(d_model, d_model, use_bias=False)


  def forward(self, x):
    B, T, C = x.shape

    emb_attn = self.w_attn(x)
    q = emb_attn[:, :, :self.d_model]
    k = emb_attn[:, :, self.d_model:self.d_model*2]
    v = emb_attn[:, :, self.d_model*2:self.d_model*3]

    # TODO: make this multihead, need to do some slicing/transposinghere

    # TODO: is * (1/x) faster than /x ??? test it out, might be interesting
    scores = (q @ k.transpose(order=(0, 2, 1))) / math.sqrt(k.shape[-1])
    # TODO: attn masks here

    scores = scores.softmax(dist_axes=(1,))

    # TODO: attn dropout

    # TODO: need to reassemble attention heads once we add those
    y = scores @ v
    # TODO: need to reassemble attention heads once we add those

    y = self.w_proj(y)

    return y

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
context_length = 8
embed_dim = 32 # 768

model = Attention(
  d_model=embed_dim,
  context_length=context_length,
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
opt = optim.Adam(model_params, learning_rate=1e-3)


# TODO: make init better because things might need to be initialized differently
# as per the GPT-2 paper
# - residual projections
# - maybe token embs???

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

  print(loss)

  loss.backward()
  opt.step()
  opt.zero_grad()
