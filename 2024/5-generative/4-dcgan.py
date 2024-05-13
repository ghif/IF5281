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

DATASET = 'cifar10'
# DATASET = 'cifar100'
# DATASET = 'celebA'
# DATASET = 'svhn'
BATCH_SIZE = 64
NUM_EPOCH = 50
IMAGE_SIZE = 64
DAY = '13may'

NC = 3 # num channels
NGPU = 0 # num gpus
NZ = 100 # num latent variables
NGF = 64 # num generator filters
NDF = 64 # num discriminator filters
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
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

if DATASET == 'cifar10':
  dataset_func = datasets.CIFAR10
elif DATASET == 'cifar100':
  dataset_func = datasets.CIFAR100
elif DATASET == 'celebA':
  dataset_func = datasets.CelebA
elif DATASET == 'svhn':
  dataset_func = datasets.SVHN

if DATASET in ['cifar10', 'cifar100']:
  # Load train
  train_data = dataset_func(
      root=DATA_DIR,
      train=True,
      download=True,
      transform=img_transform,
  )

  # Load test
  test_data = dataset_func(
      root=DATA_DIR,
      train=False,
      download=True,
      transform=img_transform,
  )
else:
  # Load train
  train_data = dataset_func(
      root=DATA_DIR,
      split='train',
      download=True,
      transform=img_transform,
  )

  # Load test
  test_data = dataset_func(
      root=DATA_DIR,
      split='test',
      download=True,
      transform=img_transform,
  )

# Create data loader
train_loader = DataLoader(
    train_data,
    shuffle=True,
    batch_size=BATCH_SIZE
)

test_loader = DataLoader(
    test_data,
    shuffle=False,
    batch_size=BATCH_SIZE
)


# Define model architecture and loss function

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc, ngpu=0):
        super().__init__()

        self.ngpu = ngpu

        self.network = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),

            # state size: (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True),

            # state size: (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            # state size: (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            # state size: (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size: (nc) x 64 x 64
        )
    
    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.network, input, range(self.ngpu))
        else:
            output = self.network(input)
        return output

netG = Generator(NZ, NGF, NC, ngpu=NGPU).to(DEVICE)
netG.apply(weights_init)

if pretrained_gen_model_path is not None:
  netG.load_state_dict(torch.load(pretrained_gen_model_path))
print(netG)

class Discriminator(nn.Module):
    def __init__(self, nz, ndf, nc, ngpu=0):
        super().__init__()
        self.ngpu = ngpu
        self.network = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # state size: (ndf * 8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.network, input, range(self.ngpu))
        else:
            output = self.network(input)
        return output.view(-1, 1).squeeze(1)        

netD = Discriminator(NZ, NDF, NC, ngpu=NGPU).to(DEVICE)
netD.apply(weights_init)
if pretrained_dis_model_path is not None:
  netD.load_state_dict(torch.load(pretrained_dis_model_path))
print(netD)

# Define loss function and optimizers
criterion = nn.BCELoss()

fixed_noise = torch.randn(64, NZ, 1, 1, device=DEVICE)
real_label = 1
fake_label = 0

optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, 0.999))
optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, 0.999))

gname = "dcgan_gen" 
gen_path = os.path.join(MODEL_DIR, f"{gname}_{DATASET}_z{NZ}_ep{NUM_EPOCH}_{DAY}.pth")

dname = "dcgan_dis" 
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
        real_cpu = X.to(DEVICE)
        batch_size = real_cpu.size(0)
        label = torch.full(
            (batch_size,), real_label,
            dtype=real_cpu.dtype, 
            device=DEVICE
        )

        output = netD(real_cpu)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.mean().item()

        # Train with fake image
        noise = torch.randn(batch_size, NZ, 1, 1, device=DEVICE)
        fake = netG(noise)
        label.fill_(fake_label)
        # output = netD(fake.detach())
        output = netD(fake)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # Update G network: minimize log(D(x)) + log(1 - D(G(z)))
        ###########################
        noise = torch.randn(batch_size, NZ, 1, 1, device=DEVICE)
        fake = netG(noise)
        label.fill_(real_label) # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        D_G_z2 = output.mean().item()

        netG.zero_grad()
        errG.backward()
        optimizerG.step()
        elapsed_t = timer.time() - start_t

        if batch_idx % 100 == 0:
            print(f'[{epoch+1}/{NUM_EPOCH}] [{batch_idx} / {len(train_loader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f} Elapsed: {elapsed_t:.2f} secs')
            vutils.save_image(real_cpu, f'{sample_dir}/real_samples.jpg', normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.detach(), f'{sample_dir}/fake_samples_epoch-{epoch+1}_batch-{batch_idx}.jpg', normalize=True)

            # Checkpointing
            torch.save(netD.state_dict(), dis_path)
            print(f" --- Discriminator model stored in {dis_path} ---")

            torch.save(netG.state_dict(), gen_path)
            print(f" --- Generator model stored in {gen_path} ---")        
    # end for batch
# end for epoch
