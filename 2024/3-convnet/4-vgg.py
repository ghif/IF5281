import os
from time import process_time
import numpy as np

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import v2

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import model_utils as mu
import train_utils as tu

# Constants
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATADIR = "/Users/mghifary/Work/Code/AI/data"
DATASET = "cifar10"
BATCH_SIZE = 128
EPOCHS = 2

# Image transformation
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

inference_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# Download training data from open datasets.
train_data = datasets.CIFAR10(
    root=DATADIR,
    train=True,
    download=True,
    transform=train_transform,
)

test_data = datasets.CIFAR10(
    root=DATADIR,
    train=False,
    download=True,
    transform=inference_transform,
)

# Create data loaders
train_dataloader = DataLoader(
    train_data, 
    batch_size=BATCH_SIZE,
    shuffle=True,
)
test_dataloader = DataLoader(
    test_data, 
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Check data loader
for X, y in train_dataloader:
    [_, c, dx1, dx2] = X.shape
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape}, {y.dtype}")
    break

num_classes = len(train_data.classes)

print(f"Number of classes: {num_classes}")

dataiter = iter(train_dataloader)
images, labels = next(dataiter)
images = images.to(DEVICE)

print(f"input shape: {images.shape}")
img_channel = images.shape[1]
print(f"output shape: {labels.shape}")

model = mu.VGG16(c, num_classes)
model = model.to(DEVICE)
print(model)
print(f"Number of params: {mu.count_parameters(model)}")
logits = model(images)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


# for epoch in range(EPOCHS):
#     losses, train_time = tu.train(model, train_dataloader, optimizer, device=DEVICE)

#     train_loss, train_acc, _ = tu.evaluate(model, train_dataloader, device=DEVICE)
#     test_loss, test_acc, _ = tu.evaluate(model, test_dataloader, device=DEVICE)

#     print(f"[Epoch {epoch + 1} / {EPOCHS}, training time: {train_time:.2f} secs] (Train) loss: {train_loss:.4f}, accuracy: {train_acc:.4f} (Test) loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")


