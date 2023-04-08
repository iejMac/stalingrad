"""
tiny shakespeare dataset

code from: https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare
"""

import os
import requests
import tiktoken
import numpy as np

# TODO: needs testing

def download_shakespeare(data_dir):
  # download the tiny shakespeare dataset
  input_file_path = os.path.join(data_dir, 'input.txt')
  if not os.path.exists(input_file_path):
      data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
      with open(input_file_path, 'w') as f:
          f.write(requests.get(data_url).text)

  with open(input_file_path, 'r') as f:
      data = f.read()
  n = len(data)
  train_data = data[:int(n*0.9)]
  val_data = data[int(n*0.9):]

  # encode with tiktoken gpt2 bpe
  enc = tiktoken.get_encoding("gpt2")
  train_ids = enc.encode_ordinary(train_data)
  val_ids = enc.encode_ordinary(val_data)
  print(f"train has {len(train_ids):,} tokens")
  print(f"val has {len(val_ids):,} tokens")

  # export to bin files
  train_ids = np.array(train_ids, dtype=np.uint16)
  val_ids = np.array(val_ids, dtype=np.uint16)
  train_ids.tofile(os.path.join(data_dir, 'train.bin'))
  val_ids.tofile(os.path.join(data_dir, 'val.bin'))
  # train.bin has 301,966 tokens
  # val.bin has 36,059 tokens

def fetch_shakespeare(data_dir="data/shakespeare"):
  train_path, val_path = os.path.join(data_dir, 'train.bin'), os.path.join(data_dir, 'val.bin')

  if not (os.path.exists(train_path) and os.path.exists(val_path)):
    download_shakespeare(data_dir)

  train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
  val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

  return train_data, val_data

def get_batch(data, block_size, batch_size):
  ix = np.random.randint(len(data) - block_size, size=batch_size)
  x = np.stack([np.array(data[i:i+block_size], dtype=np.int64) for i in ix])
  y = np.stack([np.array(data[i+1:i+1+block_size], dtype=np.int64) for i in ix])
  return x, y
