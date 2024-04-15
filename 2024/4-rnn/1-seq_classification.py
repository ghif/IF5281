"""
Original experiment from Hochreiter & Schmidhuber (1997).

The goal is to classify sequences. Elements and targets are represented locally (input vectors with only one non-zero bit). The sequence starts with an B, ends with a E (the “trigger symbol”), and otherwise consists of randomly chosen symbols from the set {a, b, c, d} except for two elements at positions t1 and t2 that are either X or Y. For the DifficultyLevel.HARD case, the sequence length is randomly chosen between 100 and 110, t1 is randomly chosen between 10 and 20, and t2 is randomly chosen between 50 and 60. There are 4 sequence classes Q, R, S, and U, which depend on the temporal order of X and Y.

The rules are:

X, X -> Q,
X, Y -> R,
Y, X -> S,
Y, Y -> U.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from time import process_time

import numpy as np
import json

from sequential_tasks import TemporalOrderExp6aSequence as QRSU

from torch import nn
from torch.nn import functional as F
import torch

from plot_lib import plot_results, set_default
import model_utils as mu

from tqdm import tqdm

# Constants
model_dir = "/Users/mghifary/Work/Code/AI/IF5281/2024/models"
device = "mps" if torch.backends.mps.is_available() else "cpu"
# device = "cpu"
max_epochs = 100
# model_type = "rnn"
model_type = "lstm"
# difficulty = "normal"
difficulty = "moderate"


    
def train(model, datagen, optimizer, device="cpu"):
    model.train()

    num_correct = 0

    num_samples = 0

    for batch_idx in range(len(datagen)):
        x, y = datagen[batch_idx]
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)

        pred = model(x)
        pred = pred[:, -1, :]

        target = torch.argmax(y, dim=1)
        loss = F.cross_entropy(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred = torch.argmax(pred, dim=1)
        num_correct += torch.sum(y_pred == target).item()

        num_samples += x.shape[0]
    # end for
    
    accuracy = num_correct / num_samples
    return num_correct, accuracy, loss.item()


def evaluate(model, datagen, device="cpu"):
    model.eval()
    num_correct = 0
    num_samples = 0 

    with torch.no_grad():
        for batch_idx in range(len(datagen)):
            x, y = datagen[batch_idx]
            x = torch.from_numpy(x).float().to(device)
            y = torch.from_numpy(y).float().to(device)
            
            pred = model(x)
            pred = pred[:, -1, :]

            target = torch.argmax(y, dim=1)
            loss = F.cross_entropy(pred, target)
            loss_val = loss.item()

            y_pred = torch.argmax(pred, dim=1)
            num_correct += torch.sum(y_pred == target).item()
            num_samples += x.shape[0]

        # end for
    accuracy = num_correct / num_samples
    return num_correct, accuracy, loss_val


# Create a dataset generator

# Create a data generator
if difficulty == "easy":
    difficulty_level = QRSU.DifficultyLevel.EASY
elif difficulty == "normal":
    difficulty_level = QRSU.DifficultyLevel.NORMAL
elif difficulty == "moderate":
    difficulty_level = QRSU.DifficultyLevel.MODERATE
elif difficulty == "hard":
    difficulty_level = QRSU.DifficultyLevel.HARD
else:
    difficulty_level = QRSU.DifficultyLevel.NIGHTMARE

example_generator = QRSU.get_predefined_generator(
    difficulty_level=difficulty_level,
    batch_size=32,
)

example_batch = example_generator[1]
print(f'The return type is a {type(example_batch)} with length {len(example_batch)}.')
print(f'The first item in the tuple is the batch of sequences with shape {example_batch[0].shape}.')
print(f'The first element in the batch of sequences is:\n {example_batch[0][0, :, :]}')
print(f'The second item in the tuple is the corresponding batch of class labels with shape {example_batch[1].shape}.')
print(f'The first element in the batch of class labels is:\n {example_batch[1][0, :]}')


# Decoding the first sequence
sequence_decoded = example_generator.decode_x(example_batch[0][0, :, :])
print(f'The sequence is: {sequence_decoded}')

# Decoding the class label of the first sequence
class_label_decoded = example_generator.decode_y(example_batch[1][0])
print(f'The class label is: {class_label_decoded}')

    
# Setup the training and test data generators
batch_size = 32
train_data_gen = QRSU.get_predefined_generator(difficulty_level, batch_size)
test_data_gen = QRSU.get_predefined_generator(difficulty_level, batch_size)  

# Setup the RNN and training settings
input_size = train_data_gen.n_symbols
# hidden_size = 8 # easy
# hidden_size = 16
hidden_size = 64 # normal, moderate
# hidden_size = 256 # moderate
# hidden_size = 512 # hard
output_size = train_data_gen.n_classes    

if model_type == "rnn":
    model = mu.SimpleRNN(
        input_size=input_size, 
        hidden_size=hidden_size,
        output_size=output_size
    )
else:
    model = mu.SimpleLSTM(
        input_size=input_size, 
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=1
    )

model = model.to(device)


# Train model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

history_train = {"loss": [], "accuracy": []}
history_test = {"loss": [], "accuracy": []}

training_time = 0.0

print(f"Using device: {device}")

with tqdm(range(max_epochs), unit="epoch") as tepoch:
    # for ep in range(max_epochs):
    for ep in tepoch:
        start_t = process_time()
        num_correct, accuracy, loss_val = train(model, train_data_gen, optimizer, device=device)
        elapsed_t = process_time() - start_t
        # print(f"[training time: {elapsed_t:.4f}] num_correct: {num_correct} out of {len(train_data_gen) * batch_size}, accuracy: {accuracy},  loss: {loss_val}")

        num_correct_test, accuracy_test, loss_val_test = evaluate(model, test_data_gen, device=device)

        history_train["loss"].append(loss_val)
        history_train["accuracy"].append(accuracy)

        history_test["loss"].append(loss_val_test)
        history_test["accuracy"].append(accuracy_test)

        # print(f"[Epoch {ep + 1} / {max_epochs} - train time: {elapsed_t: .2f} secs] (Train) loss: {loss_val:.4f}, accuracy: {accuracy:.2f}, (Test) loss: {loss_val_test:.4f}, accuracy: {accuracy_test:.2f}")

        training_time += elapsed_t

        tepoch.set_postfix({
            "training_time": elapsed_t,
            "train_loss": loss_val,
            "train_accuracy": accuracy,
            "test_loss": loss_val_test,
            "test_accuracy": accuracy_test,
        })

# end for

print(f"Training finished with total time: {training_time:.2f} secs")

# Save model
model_name = f"{model_type}_qrsu-{difficulty}"
model_path = os.path.join(model_dir, f"{model_name}.pth")
torch.save(model.state_dict(), model_path)

# Save training history
history_path = os.path.join(model_dir, f"{model_name}_hist.json")
history = {
    "train": history_train,
    "test": history_test,
}

json_object = json.dumps(history) # serializing json
with open(history_path, "w") as outfile:
    outfile.write(json_object) # write to json file
    
print(f"Saved PyTorch Model State to {model_path} and history xto {history_path}")

# Visualize training results
set_default(figsize=(20, 10))

plot_results(
    history_train["loss"], 
    history_test["loss"], 
    xlabel="Epoch",
    ylabel="Loss",
    legend=["Train", "Test"],
    title=model_type,
    figpath=f"{model_type}_qrsu-{difficulty}_loss.png"
)

plot_results(
    history_train["accuracy"], 
    history_test["accuracy"], 
    xlabel="Epoch",
    ylabel="Accuracy",
    ylim=1.0,
    legend=["Train", "Test"],
    title=model_type,
    figpath=f"{model_type}_qrsu-{difficulty}_accuracy.png"
)