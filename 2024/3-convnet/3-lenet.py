import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np
from time import process_time

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from plot_lib import plot_data, plot_model, plot_results, set_default

import train_utils as tu
import model_utils as mu

# Constants
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
# DEVICE = "cpu"
DATADIR = "/Users/mghifary/Work/Code/AI/data"
BATCH_SIZE = 128

EPOCHS = 2


# Define image transformation
train_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

inference_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Download training data from open datasets.
train_data = datasets.FashionMNIST(
    root=DATADIR,
    train=True,
    download=True,
    transform=train_transform
)


test_data = datasets.FashionMNIST(
    root=DATADIR,
    train=False,
    download=True,
    transform=inference_transform
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

for images, labels in train_dataloader:
    [_, c, dx1, dx2] = images.shape
    print(f"Shape of X [N, C, H, W]: {images.shape}")
    print(f"Shape of y: {labels.shape}, {labels.dtype}")
    break


# # helper function to show an image
# # (used in the `plot_classes_preds` function below)
# def matplotlib_imshow(img, one_channel=False):
#     if one_channel:
#         img = img.mean(dim=0)
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     if one_channel:
#         plt.imshow(npimg, cmap="Greys")
#     else:
#         plt.imshow(np.transpose(npimg, (1, 2, 0)))

# # get some random training images
# dataiter = iter(train_dataloader)
# images, labels = next(dataiter)

# # create grid of images
# img_grid = torchvision.utils.make_grid(images)

# # show images
# matplotlib_imshow(img_grid, one_channel=True)


num_classes = len(train_data.classes)

model = mu.LeNet5(c, num_classes)
model = model.to(DEVICE)
print(model)

# images = images.to(DEVICE)
# logits = model(images)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


for epoch in range(EPOCHS):
    losses, train_time = tu.train(model, train_dataloader, optimizer, device=DEVICE)

    train_loss, train_acc, _ = tu.evaluate(model, train_dataloader, device=DEVICE)
    test_loss, test_acc, _ = tu.evaluate(model, test_dataloader, device=DEVICE)

    print(f"[Epoch {epoch + 1} / {EPOCHS}, training time: {train_time:.2f} secs] (Train) loss: {train_loss:.4f}, accuracy: {train_acc:.4f} (Test) loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")
