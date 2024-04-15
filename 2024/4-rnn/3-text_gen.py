import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from time import process_time

import seq_processor as sp

import torch
from torch import nn
from torch.nn import functional as F

class SimpleBigram(nn.Module):
    def __init__(self, vocab_size, seq_len, n_embed, n_hidden, num_layers=1):
        super().__init__()
        self.seq_len = seq_len

        self.embedding_table = nn.Embedding(vocab_size, n_embed)

        self.lstm = nn.LSTM(n_embed, n_hidden, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(n_hidden, vocab_size)
    
    def forward(self, idx):
        x = self.embedding_table(idx)
        h, _ = self.lstm(x)
        logits = self.linear(h)
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens/characters given an input

        Args:
            idx (torch.tensor): (B, T) array of indices in the current context
            max_new_tokens (int): maximum number of new tokens to generate
        Returns:
            idx (torch.tensor): (B, T+max_new_tokens) array of indices in the current context
        """
        self.eval()
        
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.seq_len:]
            # Predict
            # logits, loss = self(idx)
            logits = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        self.train()    
        return idx


@torch.no_grad()
def estimate_loss(
        model, 
        data, 
        eval_iters=10,
        batch_size=32,
        seq_len=64,
        device="cpu"
    ):
    
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xb, yb = sp.get_batch(data, batch_size=batch_size, block_size=seq_len)
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses[k] = loss.item()

    avg_loss = losses.mean()
    model.train()

    return avg_loss

def loss_fn(logits, y):
    B = y.size(0)
    T = y.size(1)
    pred = logits.view(B*T, -1)
    groundtruth = y.view(B*T)
    return F.cross_entropy(pred, groundtruth)

# Constants
model_dir = "/Users/mghifary/Work/Code/AI/IF5281/2024/models"
device = "mps" if torch.backends.mps.is_available() else "cpu"
max_iters = 5000
data_dir = "/Users/mghifary/Work/Code/AI/IF5281/2024/data"
data_path = os.path.join(data_dir, "chairilanwar.txt")
seq_len = 256
n_embed = 384
n_hidden = 512
batch_size = 64
eval_interval = 5

# Load sequence data
chproc = sp.CharProcessor(data_path)
data = torch.tensor(chproc.encode(chproc.text), dtype=torch.long)

# Construct training data
n = len(data)
train_data = data[:n]
val_data = None

# Print pairs of input and target
x = train_data[:seq_len]
y = train_data[1:seq_len+1]
for t in range(seq_len):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context}, the target is {target}")

torch.manual_seed(1337)

model = SimpleBigram(
    chproc.vocab_size,
    seq_len,
    n_embed,
    n_hidden,
    num_layers=1
)

model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f"Using device: {device}")

idx = torch.zeros((1, 1), dtype=torch.long, device=device)

for step in range(max_iters):
    xb, yb = sp.get_batch(train_data, 
        batch_size=batch_size, 
        block_size=seq_len
    )

    xb = xb.to(device)
    yb = yb.to(device)

    start_t = process_time()
    # Forward pass
    logits = model(xb)

    # Compute loss
    loss = loss_fn(logits, yb)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    elapsed_t = process_time() - start_t

    if step % eval_interval == 0:
        train_loss = estimate_loss(model, train_data, eval_iters=10, batch_size=batch_size, seq_len=seq_len, device=device)

        print(f"[Step-{step+1}/{max_iters} - training time: {elapsed_t:.3f} secs]: train loss: {train_loss:.4f}")

        pred_idx = model.generate(idx, 100)
        pred_str = chproc.decode(pred_idx[0].tolist())

        # print(f"pred_idx: {pred_idx}")
        print(f"pred_str: {pred_str}")
        print("\n")

        # Save model
        model_name = f"bigram_lstm_chairilanwar"
        model_path = os.path.join(model_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
# end for

