import os
import jax
import jax.numpy as jnp

class CharProcessor:
    def __init__(self, datapath):
        # Read the data
        with open(datapath, "r", encoding="utf-8") as f:
            self.text = f.read()

        print(f"Length of text: {len(self.text)} characters")

        # The unique characters in the file
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        print("".join(self.chars))
        print(self.vocab_size)

        # Create a mapping from characters to integers
        self.stoi = { ch:i for i, ch in enumerate(self.chars) }
        self.itos = { i:ch for i, ch in enumerate(self.chars) }
        self.encode = lambda s: [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
        self.decode = lambda l: "".join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string

def get_batch(data, batch_size=4, block_size=8, key=None):
    """
    Generate a batch of data of inputs x and target y

    Args:
        data (Array): (N,) full encoded text in integers
        batch_size (int): number of samples in a batch
        block_size (int): number of time steps in a sequence
        key (PRNGKey): JAX random key
    
    Returns:
        x (Array): (batch_size, block_size) input sequence
        y (Array): (batch_size, block_size) output sequence
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    ix = jax.random.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[t:t+block_size] for t in ix])
    y = jnp.stack([data[t+1:t+block_size+1] for t in ix])
    return x, y
