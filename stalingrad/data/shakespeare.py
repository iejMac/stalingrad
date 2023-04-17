"""
tiny shakespeare dataset

code from: https://github.com/karpathy/nanoGPT/tree/master/data/shakespeare
"""
import os
import pickle
import requests
import tiktoken
import numpy as np

# TODO: needs testing
def download_shakespeare(data_dir, tokenizer="gpt2"):
  # download the tiny shakespeare dataset
  input_file_path = os.path.join(data_dir, 'input.txt')
  print(input_file_path)
  if not os.path.exists(input_file_path):
      data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
      with open(input_file_path, 'w') as f:
          f.write(requests.get(data_url).text)

  with open(input_file_path, 'r') as f:
      data = f.read()
  n = len(data)

  if tokenizer == "gpt2":
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    # encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")
  else:  # shakespeare_char
    # get all the unique characters that occur in this text
    print(f"length of dataset in characters: {len(data):,}")
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    def encode(s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # create the train and test splits
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]

    # encode both to integers
    train_ids = encode(train_data)
    val_ids = encode(val_data)

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(data_dir, 'meta_char.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
  print(f"train has {len(train_ids):,} tokens")
  print(f"val has {len(val_ids):,} tokens")

  # export to bin files
  train_ids = np.array(train_ids, dtype=np.uint16)
  val_ids = np.array(val_ids, dtype=np.uint16)
  train_ids.tofile(os.path.join(data_dir, f'train_{tokenizer}.bin'))
  val_ids.tofile(os.path.join(data_dir, f'val_{tokenizer}.bin'))
  # train.bin has 301,966 tokens
  # val.bin has 36,059 tokens

def fetch_shakespeare(data_dir="data/shakespeare", tokenizer="gpt2"):
  train_path, val_path = os.path.join(data_dir, f'train_{tokenizer}.bin'), os.path.join(data_dir, f'val_{tokenizer}.bin')

  if not (os.path.exists(train_path) and os.path.exists(val_path)):
    download_shakespeare(data_dir, tokenizer=tokenizer)

  train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
  val_data = np.memmap(val_path, dtype=np.uint16, mode='r')

  return train_data, val_data

def get_batch(data, block_size, batch_size):
  ix = np.random.randint(len(data) - block_size, size=batch_size)
  x = np.stack([np.array(data[i:i+block_size], dtype=np.int64) for i in ix])
  y = np.stack([np.array(data[i+1:i+1+block_size], dtype=np.int64) for i in ix])
  return x, y
