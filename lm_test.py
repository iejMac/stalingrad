import numpy as np
import tiktoken

from stalingrad.tensor import Tensor
from stalingrad import nn
from stalingrad.data import fetch_shakespeare, get_batch

SEQ_LEN = 4
EMB_DIM = 8


'''
# s-attn

tok_sequence = Tensor(np.zeros((SEQ_LEN, EMB_DIM)))
tok_sequence[0, 0] = 1
tok_sequence[1, 3] = 1
tok_sequence[2, 7] = 1
tok_sequence[3, 1] = 1


w_keys = nn.Linear(EMB_DIM, EMB_DIM, use_bias=False)
w_queries = nn.Linear(EMB_DIM, EMB_DIM, use_bias=False)
w_values = nn.Linear(EMB_DIM, EMB_DIM, use_bias=False)

keys = w_keys(tok_sequence) # (SEQ_LEN, EMB_DIM)
queries = w_queries(tok_sequence) # (SEQ_LEN, EMB_DIM)
values = w_values(tok_sequence) # (SEQ_LEN, EMB_DIM)

scores = keys @ queries.transpose()
norm_scores = scores / scores.sum(axis=1).reshape(shape=(-1, 1))

contextualized_sequence = norm_scores @ values

print(contextualized_sequence)
'''

train_data, val_data = fetch_shakespeare(data_dir="data/shakespeare")

# training stuff
steps = 10

batch_size = 4
context_length = 8


for step in range(steps):
  bx, by = get_batch(train_data, context_length, batch_size)
  tbx, tby = Tensor(bx), Tensor(by)

  print(tbx.shape, tby.shape)








