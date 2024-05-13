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
NUM_EPOCH = 50
HIDDEN_SIZE = 40

NVIZ = 512
nrow = np.floor(np.sqrt(NVIZ)).astype(int)

DATASET = "mnist"
DAY = '11may'
BETA = 1
MNAME = "vae"

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

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
class VariationalAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size ** 2),
            nn.Tanh(),
            nn.Linear(hidden_size ** 2, hidden_size * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size ** 2),
            nn.Tanh(),
            nn.Linear(hidden_size ** 2, input_size),
            nn.Tanh(),
        )
    
    def reparameterise(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new_empty(std.size()).normal_()
            return eps.mul_(std).add_(mu)
        else:
            return mu
    
    def encode(self, x):
        mu_logvar = self.encoder(x)
        mu_logvar = mu_logvar.view(-1, 2, self.hidden_size)
        mu = mu_logvar[:, 0, :]
        logvar = mu_logvar[:, 1, :]
        z = self.reparameterise(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z, mu, logvar = self.encode(x)
        xr = self.decode(z)
        return xr, mu, logvar

def vae_loss(x_hat, x, mu, logvar, beta=1):
    rec_loss = F.mse_loss(x_hat, x, reduction="sum")
    kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)

    vae_loss = rec_loss + beta * kl_loss
    return vae_loss, rec_loss, kl_loss

# Set training configuration
[n, dx1, dx2] = train_data.data.size()
ch = 1
input_size = dx1 * dx2

model = VariationalAutoencoder(input_size, HIDDEN_SIZE).to(DEVICE)

optimizer = torch.optim.AdamW(
    lr=3e-4,
    params=model.parameters()
)

model_path = os.path.join(MODEL_DIR, f"{MNAME}_mnist_z{HIDDEN_SIZE}_ep{NUM_EPOCH}.pth")
# create sample_dir if not exists
fname, ext = os.path.splitext(model_path)

sample_dir = os.path.join(MODEL_DIR, f"{fname}_{DAY}")
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
    print(f'The new directory {sample_dir} has been created')


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
vutils.save_image(in_imgs.detach(), f'{sample_dir}/real_samples.jpg', normalize=True, nrow=nrow)

in_flatten = torch.flatten(in_imgs, start_dim=1)
labels = torch.cat(labels, dim=0)

fixed_latent = torch.randn(64, HIDDEN_SIZE, device=DEVICE)

# Train model

for epoch in range(NUM_EPOCH):
    start_t = timer.time()
    with tqdm(train_loader, unit="batch") as tepoch:
        for batch_idx, (X, _) in enumerate(tepoch):
            model.train()

            # Feed forward / reconstruct
            X = torch.flatten(X, start_dim=1)
            X = X.to(DEVICE)
            Xr, Mu, Logvar = model(X)
            
            # Compute loss
            total_loss, rec_loss, kl_loss = vae_loss(Xr, X, Mu, Logvar, beta=BETA)

            # Backprop
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            model.eval()

        
        # end for
    # end with
    elapsed_t = timer.time() - start_t

    print(f'Epoch [{epoch+1}/{NUM_EPOCH}], total_loss: {total_loss.item():.4f}, rec_loss: {rec_loss.item():.4f}, kl_loss: {kl_loss.item():.4f}, elapsed_t: {elapsed_t: 0.2f} secs')

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f" ---- Model {model_path} stored!")

    # Display input images and their reconstructions
    with torch.no_grad():
        rec_imgs, _, _ = model(in_flatten)

    rec_imgs = rec_imgs.view(-1, ch, dx1, dx2)
    vutils.save_image(rec_imgs.detach(), f'{sample_dir}/reconstructed_samples_{epoch}.jpg', normalize=True, nrow=nrow)

    # Display reconstruction from fixed, random latent variables
    with torch.no_grad():
        fixed_rec_imgs = model.decode(fixed_latent)

    fixed_rec_imgs = fixed_rec_imgs.view(-1, ch, dx1, dx2)
    vutils.save_image(fixed_rec_imgs.detach(), f'{sample_dir}/fixed_rec_{epoch}.jpg', normalize=True, nrow=8)

    # Visualize encoded features with t-SNE
    # Get encoded features
    with torch.no_grad():
        _, Mu, _ = model.encode(in_flatten)

    feat = Mu.detach().cpu().numpy()
    tsne_path = os.path.join(sample_dir, f"tsne_{epoch}.jpg")
    vu.plot_features_tsne(feat, labels, tsne_path)

# end for

