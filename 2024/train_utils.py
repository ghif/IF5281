from time import process_time
from tqdm import tqdm

import torch
import torch.nn.functional as F


# Define training and inference methods
def train(model, dataloader, optimizer, device="cpu"):
    print(f"Training -- using device {device}")
    train_time = 0.0
    losses = 0.
    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, y) in tepoch:
            start_t = process_time()
            X, y = X.to(device), y.to(device)
            # print(f"X shape: {X.shape}")
            # Compute prediction error
            logits = model(X)

            # Loss function
            loss = F.cross_entropy(logits, y)
            batch_size = X.shape[0]
            losses += (loss.item() / batch_size)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            elapsed_time = process_time() - start_t
            train_time += elapsed_time

            # print(f"[Batch-{b}] loss: {loss.item()}, training time (secs): {train_time}")

            tepoch.set_postfix({
                'loss': loss.item(),
                'time(secs)': train_time
            })
        # end for
            
    num_data = len(dataloader.dataset)
    losses /= num_data
            
    return losses, train_time

def evaluate(model, dataloader, device="cpu"):
    print(f"Evaluation -- using device {device}")

    model.eval()
    eval_time = 0.0
    losses = 0.0
    correct = 0
    
    with tqdm(dataloader, unit="batch") as tepoch:
        for (X, y) in tepoch:
            start_t = process_time()
            X, y = X.to(device), y.to(device)
            
            # Compute loss
            with torch.no_grad():
                logits = model(X)
            loss = F.cross_entropy(logits, y)
            batch_size = X.shape[0]
            losses += (loss.item() / batch_size)

            # Compute correct predictions
            pred = torch.argmax(logits, axis=1)
            correct += torch.sum(pred == y)

            elapsed_time = process_time() - start_t
            eval_time += elapsed_time


            tepoch.set_postfix({
                'loss': loss.item(),
                'time(secs)': eval_time
            })
        # end for
            
    model.train()

    num_data = len(dataloader.dataset)
    accuracy = correct.item() / num_data
    return losses, accuracy, eval_time