import torch 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision import datasets
import torchvision.utils as vutils

from skimage.util import random_noise

import sys, os
import numpy as np
import time as timer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import viz_utils as vu

# Define constants
DATA_DIR = "/Users/mghifary/Work/Code/AI/data"
MODEL_DIR = "/Users/mghifary/Work/Code/AI/IF5281/2024/models"

BATCH_SIZE = 128
NUM_EPOCH = 20
IS_DENOISING = True # True: Denoising Autoencoder, False: Standard Autoencoder
# NOISE_TYPE = "gaussian" # {"gaussian", "salt"}
NOISE_TYPE = "gaussian" # {"gaussian", "salt"}
NVIZ = 512
nrow = np.floor(np.sqrt(NVIZ)).astype(int)

DATASET = "mnist"
DAY = '13may_v2'

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

if not IS_DENOISING:
    SAMPLE_DIR = os.path.join(MODEL_DIR, f"ae_samples_{DATASET}_{DAY}")
else:
    SAMPLE_DIR = os.path.join(MODEL_DIR, f"dae_samples_{NOISE_TYPE}_{DATASET}_{DAY}")

# create SAMPLE_DIR if not exists
if not os.path.exists(SAMPLE_DIR):
    os.makedirs(SAMPLE_DIR)
    print(f'The new directory {SAMPLE_DIR} has been created')

# Load training and test data
# Transform to (-1, 1) 
img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

# Load train
train_data = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=img_transform,
)

# Load test
test_data = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=img_transform,
)

# Create data loader
train_loader = DataLoader(
    train_data,
    shuffle=True,
    batch_size=BATCH_SIZE,
)

test_loader = DataLoader(
    test_data,
    shuffle=False,
    batch_size=BATCH_SIZE,
)

# Define model architecture
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Tanh(),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y


# Set training configuration
[n, dx1, dx2] = train_data.data.size()
ch = 1
input_size = dx1 * dx2
hidden_size = 128
model = Autoencoder(input_size, hidden_size).to(DEVICE)

optimizer = torch.optim.Adam(
    lr=3e-4,
    params=model.parameters()
)

mname = "ae" if not IS_DENOISING else f"dae_{NOISE_TYPE}"
model_path = os.path.join(MODEL_DIR, f"{mname}_mnist_z{hidden_size}_ep{NUM_EPOCH}.pth")

# Get fixed input test image
in_imgs = []
labels = []
nsamples = 0
print(f"Get fixed input for testing visualization ...")
for batch_idx, (Xt, Yt) in enumerate(test_loader):
    ns = Xt.shape[0]
    nsamples += ns
    print(f'Batch {batch_idx}, nsamples: {nsamples}')
    in_imgs.append(Xt)
    labels.append(Yt)

    if nsamples >= NVIZ:
        break
    
in_imgs = torch.cat(in_imgs, dim=0)
in_imgs = in_imgs.to(DEVICE)
labels = torch.cat(labels, dim=0)

vutils.save_image(in_imgs.detach(), f'{SAMPLE_DIR}/real_samples.jpg', normalize=True, nrow=nrow)

# Train model
for epoch in range(NUM_EPOCH):
    start_t = timer.time()
    # for batch_idx, (X, _) in enumerate(train_loader):
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (X, _) in enumerate(tepoch):
            if IS_DENOISING:
                # Add noise
                # noise = torch.randn_like(X) * 0.2
                # Xn = X + noise
                if NOISE_TYPE == "salt":
                    Xn = torch.tensor(random_noise(X, mode=NOISE_TYPE, amount=0.2))
                    # print(f'[Salt-and-Pepper Noise] Xn dtype: {Xn.dtype}')
                elif NOISE_TYPE == "gaussian":
                    Xn = torch.tensor(random_noise(X, mode=NOISE_TYPE, var=0.5)).to(torch.float32)
                    # print(f'[Gaussian Noise] Xn dtype: {Xn.dtype}')
            else:
                Xn = X

            # Feed forward
            X = torch.flatten(X, start_dim=1).to(DEVICE)
            Xn = torch.flatten(Xn, start_dim=1).to(DEVICE)
            Xr = model(Xn)

            # Compute loss
            loss = F.mse_loss(Xr, X)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        # end for
    # end with
    elapsed_t = timer.time() - start_t

    print(f'Epoch [{epoch+1}/{NUM_EPOCH}], loss: {loss.item():.4f}, elapsed_t: {elapsed_t: 0.2f} secs')

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f" ---- Model {model_path} stored!")

    # Display input images and their reconstructions
    in_flatten = torch.flatten(in_imgs, start_dim=1)
    with torch.no_grad():
        rec_imgs = model(in_flatten)
        
    rec_imgs = rec_imgs.view(-1, ch, dx1, dx2)
    vutils.save_image(rec_imgs.detach(), f'{SAMPLE_DIR}/reconstructed_samples_{epoch}.jpg', normalize=True, nrow=nrow)
    
    if IS_DENOISING:
        if NOISE_TYPE == "salt":
            in_imgs_n = torch.tensor(random_noise(in_imgs.detach().cpu().numpy(), mode=NOISE_TYPE, amount=0.2))
        elif NOISE_TYPE == "gaussian":
            in_imgs_n = torch.tensor(random_noise(in_imgs.detach().cpu().numpy(), mode=NOISE_TYPE, var=0.5)).to(torch.float32)
        
        vutils.save_image(in_imgs_n, f'{SAMPLE_DIR}/noisy_samples.jpg', normalize=True, nrow=nrow)


    # Visualize encoded features with t-SNE
    # Get encoded features
    with torch.no_grad():
        Z = model.encoder(in_flatten)

    feat = Z.detach().cpu().numpy()
    tsne_path = os.path.join(SAMPLE_DIR, f"tsne_{epoch}.jpg")
    vu.plot_features_tsne(feat, labels, tsne_path)

# end for

