import torch 
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision import datasets
import torchvision.utils as vutils


import sys, os
import time as timer
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import viz_utils as vu

# Define constants
DATA_DIR = "/Users/mghifary/Work/Code/AI/data"
MODEL_DIR = "/Users/mghifary/Work/Code/AI/IF5281/2024/models"

# Define default constants
DATASET = 'mnist'
DAY = '13may'
BATCH_SIZE = 64
NUM_EPOCH = 50


NC = 1 # num channels
NZ = 100 # num latent variables
LR = 2e-4 # learning rate
BETA1 = 0.5 # beta1 for Adam optimizer

# pretrained_gen_model_path = os.path.join(MODEL_DIR, f"dcgan_gen_{DATASET}_z100_ep25.pth")
# pretrained_dis_model_path = os.path.join(MODEL_DIR, f"dcgan_dis_{DATASET}_z100_ep25.pth")
pretrained_gen_model_path = None
pretrained_dis_model_path = None

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
# Define model architecture and loss function

# custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm") != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.zeros_(m.bias.data)

class Generator(nn.Module):
    """
        Simple Generator w/ MLP
    """
    def __init__(self, input_size=100, output_size=784):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_size),
            nn.Tanh()
        )
        
    def forward(self, z):
        x = self.layer(z)
        x = x.view(x.size(0), NC, 28, 28)
        return x
    
# Set training configugration
[n, dx1, dx2] = train_data.data.size()
ch = 1
input_size = dx1 * dx2

netG = Generator(input_size=NZ, output_size=dx1 * dx2).to(DEVICE)
# netG.apply(weights_init)

if pretrained_gen_model_path is not None:
  netG.load_state_dict(torch.load(pretrained_gen_model_path))
print(netG)

class Discriminator(nn.Module):
    """
        Simple Discriminator w/ MLP
    """
    def __init__(self, input_size=784, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, num_classes),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        y = self.layer(x)
        y = torch.flatten(y)
        return y

netD = Discriminator(input_size=dx1 * dx2, num_classes=1).to(DEVICE)
# netD.apply(weights_init)
if pretrained_dis_model_path is not None:
  netD.load_state_dict(torch.load(pretrained_dis_model_path))
print(netD)

# Define loss function and optimizers
criterion = nn.BCELoss()

# fixed_noise = torch.randn(64, NZ, 1, 1, device=DEVICE)
fixed_latent = torch.randn(BATCH_SIZE, NZ, device=DEVICE)
real_label = 1
fake_label = 0

optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

gname = "gan_gen" 
gen_path = os.path.join(MODEL_DIR, f"{gname}_{DATASET}_z{NZ}_ep{NUM_EPOCH}_{DAY}.pth")

dname = "gan_dis" 
dis_path = os.path.join(MODEL_DIR, f"{dname}_{DATASET}_z{NZ}_ep{NUM_EPOCH}_{DAY}.pth")

fname, ext = os.path.splitext(gen_path)

sample_dir = os.path.join(MODEL_DIR, f"{fname}")
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
    print(f'The new directory {sample_dir} has been created')

for epoch in range(NUM_EPOCH):
    
    for batch_idx, (X, _) in enumerate(train_loader):
        start_t = timer.time()
        ############################
        #  Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################

        # Train with real image
        
        netD.zero_grad()

        real_x = X.to(DEVICE)
        real_x = torch.flatten(real_x, start_dim=1)
        batch_size = real_x.size(0)
        label = torch.full(
            (batch_size,), real_label,
            dtype=real_x.dtype, 
            device=DEVICE
        )
        outputD = netD(real_x)
        D_x = outputD.mean().item()
        errD_real = criterion(outputD, label)
        errD_real.backward()
        

        # Train with fake image
        noisy_latent = torch.randn(batch_size, NZ, device=DEVICE)
        fake_x = netG(noisy_latent)
        label.fill_(fake_label)
        fake_x = torch.flatten(fake_x, start_dim=1)
        outputD = netD(fake_x)
        D_G_z1 = outputD.mean().item()
        errD_fake = criterion(outputD, label)
        errD_fake.backward()

        errD = errD_real + errD_fake

        # netD.zero_grad()
        # errD.backward()
        optimizerD.step()

        ############################
        # Update G network: minimize log(D(x)) + log(1 - D(G(z)))
        ###########################

        noisy_latent = torch.randn(batch_size, NZ, device=DEVICE)
        fake_x = netG(noisy_latent)
        label.fill_(real_label) # assign fake labels as real labels for generator cost
        
        fake_x = torch.flatten(fake_x, start_dim=1)
        outputD = netD(fake_x)
        D_G_z2 = outputD.mean().item()

        errG = criterion(outputD, label)
        
        netG.zero_grad()
        errG.backward()
        optimizerG.step()

        elapsed_t = timer.time() - start_t

        if batch_idx % 100 == 0:
            print(f'[{epoch+1}/{NUM_EPOCH}] [{batch_idx} / {len(train_loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f} Elapsed: {elapsed_t:.2f} secs')
            vutils.save_image(real_x, f'{sample_dir}/real_samples.jpg', normalize=True)
            fixed_fake = netG(fixed_latent)
            vutils.save_image(fixed_fake.detach(), f'{sample_dir}/fake_samples_epoch-{epoch+1}_batch-{batch_idx}.jpg', normalize=True)

            # Checkpointing
            torch.save(netD.state_dict(), dis_path)
            print(f" --- Discriminator model stored in {dis_path} ---")

            torch.save(netG.state_dict(), gen_path)
            print(f" --- Generator model stored in {gen_path} ---")        
    # end for batch
# end for epoch
