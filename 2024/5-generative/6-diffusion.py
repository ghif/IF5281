from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
import torch
import torchvision.utils as vutils

import math
from einops import parse_shape, rearrange

from sklearn.datasets import fetch_openml
import numpy as np

import torch.optim as optim
import logging

import os

from time import process_time
# ITERATIONS = 2000
# BATCH_SIZE = 512

DATASET = "mnist"
MODEL_DIR = "/Users/mghifary/Work/Code/AI/IF5281/2024/models"
DAY = "22may"

BATCH_SIZE = 256
ITERATIONS = 2000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def unsqueeze_to(tensor, target_ndim):
    assert tensor.ndim <= target_ndim
    while tensor.ndim < target_ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor

def unsqueeze_as(tensor, target_tensor):
    assert tensor.ndim <= target_tensor.ndim
    while tensor.ndim < target_tensor.ndim:
        tensor = tensor.unsqueeze(-1)
    return tensor

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_length=10000):
        super().__init__()
        self.register_buffer("embedding", self.make_embedding(dim, max_length))

    def forward(self, x):
        # Parameters
        #   x: (bsz,) discrete
        return self.embedding[x]

    @staticmethod
    def make_embedding(dim, max_length=10000):
        embedding = torch.zeros(max_length, dim)
        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(max_length / 2 / math.pi) / dim))
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding

class SelfAttention2d(nn.Module):
    """
    Only implements the MultiHeadAttention component, not the PositionwiseFFN component.
    """
    def __init__(self, dim, num_heads=8, dropout_prob=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.q_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.k_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.v_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.o_conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        q = self.q_conv(x)
        k = self.k_conv(x)
        v = self.v_conv(x)
        q = rearrange(q, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        k = rearrange(k, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        v = rearrange(v, "b (g c) h w -> (b g) c (h w)", g=self.num_heads)
        a = torch.einsum("b c s, b c t -> b s t", q, k) / self.dim ** 0.5
        a = self.dropout(torch.softmax(a, dim=-1))
        o = torch.einsum("b s t, b c t -> b c s", a, v)
        o = rearrange(o, "(b g) c (h w) -> b (g c) h w", g=self.num_heads, w=x.shape[-1])
        return x + self.o_conv(o)
    
class BasicBlock(nn.Module):
    """
    BasicBlock: two 3x3 convs followed by a residual connection then ReLU.
    [He et al. CVPR 2016]

        BasicBlock(x) = ReLU( x + Conv3x3( ReLU( Conv3x3(x) ) ) )

    This version supports an additive shift parameterized by time.
    """
    def __init__(self, in_c, out_c, time_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.mlp_time = nn.Sequential(
            nn.Linear(time_c, time_c),
            nn.ReLU(),
            nn.Linear(time_c, out_c),
        )
        if in_c == out_c:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x, t):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out + unsqueeze_as(self.mlp_time(t), x))
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out + self.shortcut(x))
        return out
    
class UNet(nn.Module):
    """
    Simple implementation that closely mimics the one by Phil Wang (lucidrains).
    """
    def __init__(self, in_dim, embed_dim, dim_scales):
        super().__init__()

        self.init_embed = nn.Conv2d(in_dim, embed_dim, 1)
        self.time_embed = PositionalEmbedding(embed_dim)

        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()

        # Example:
        #   in_dim=1, embed_dim=32, dim_scales=(1, 2, 4, 8) => all_dims=(32, 32, 64, 128, 256)
        all_dims = (embed_dim, *[embed_dim * s for s in dim_scales])

        for idx, (in_c, out_c) in enumerate(zip(
            all_dims[:-1],
            all_dims[1:],
        )):
            is_last = idx == len(all_dims) - 2
            self.down_blocks.extend(nn.ModuleList([
                BasicBlock(in_c, in_c, embed_dim),
                BasicBlock(in_c, in_c, embed_dim),
                nn.Conv2d(in_c, out_c, 3, 2, 1) if not is_last else nn.Conv2d(in_c, out_c, 1),
            ]))

        for idx, (in_c, out_c, skip_c) in enumerate(zip(
            all_dims[::-1][:-1],
            all_dims[::-1][1:],
            all_dims[:-1][::-1],
        )):
            is_last = idx == len(all_dims) - 2
            self.up_blocks.extend(nn.ModuleList([
                BasicBlock(in_c + skip_c, in_c, embed_dim),
                BasicBlock(in_c + skip_c, in_c, embed_dim),
                nn.ConvTranspose2d(in_c, out_c, 2, 2) if not is_last else nn.Conv2d(in_c, out_c, 1),
            ]))

        self.mid_blocks = nn.ModuleList([
            BasicBlock(all_dims[-1], all_dims[-1], embed_dim),
            SelfAttention2d(all_dims[-1]),
            BasicBlock(all_dims[-1], all_dims[-1], embed_dim),
        ])
        self.out_blocks = nn.ModuleList([
            BasicBlock(embed_dim, embed_dim, embed_dim),
            nn.Conv2d(embed_dim, in_dim, 1, bias=True),
        ])

    def forward(self, x, t):
        x = self.init_embed(x)
        t = self.time_embed(t)
        skip_conns = []
        residual = x.clone()

        for block in self.down_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t)
                skip_conns.append(x)
            else:
                x = block(x)
        for block in self.mid_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t)
            else:
                x = block(x)
        for block in self.up_blocks:
            if isinstance(block, BasicBlock):
                x = torch.cat((x, skip_conns.pop()), dim=1)
                x = block(x, t)
            else:
                x = block(x)

        x = x + residual
        for block in self.out_blocks:
            if isinstance(block, BasicBlock):
                x = block(x, t)
            else:
                x = block(x)
        return x

@dataclass(frozen=True)
class DiffusionModelConfig:
    num_timesteps: int
    target_type: str = "pred_eps"
    noise_schedule_type: str = "cosine"
    loss_type: str = "l2"
    gamma_type: float = "ddim"

    def __post_init__(self):
        assert self.num_timesteps > 0
        assert self.target_type in ("pred_x_0", "pred_eps", "pred_v")
        assert self.noise_schedule_type in ("linear", "cosine")
        assert self.loss_type in ("l1", "l2")
        assert self.gamma_type in ("ddim", "ddpm")

class DiffusionModel(nn.Module):

    def __init__(
        self,
        input_shape: tuple[int, ...],
        nn_module: nn.Module,
        config: DiffusionModelConfig,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.nn_module = nn_module
        self.num_timesteps = config.num_timesteps
        self.target_type = config.target_type
        self.gamma_type = config.gamma_type
        self.noise_schedule_type = config.noise_schedule_type
        self.loss_type = config.loss_type

        # Input shape must be either (c,) or (c, h, w) or (c, t, h, w)
        assert len(input_shape) in (1, 3, 4)

        # Construct the noise schedule
        if self.noise_schedule_type == "linear":
            beta_t = torch.linspace(1e-4, 2e-2, self.num_timesteps + 1)
            alpha_t = torch.cumprod(1 - beta_t, dim=0) ** 0.5
        elif self.noise_schedule_type == "cosine":
            linspace = torch.linspace(0, 1, self.num_timesteps + 1)
            f_t = torch.cos((linspace + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
            bar_alpha_t = f_t / f_t[0]
            beta_t = torch.zeros_like(bar_alpha_t)
            beta_t[1:] = (1 - (bar_alpha_t[1:] / bar_alpha_t[:-1])).clamp(min=0, max=0.999)
            alpha_t = torch.cumprod(1 - beta_t, dim=0) ** 0.5
        else:
            raise AssertionError(f"Invalid {self.noise_schedule_type=}.")

        # These tensors are shape (num_timesteps + 1, *self.input_shape)
        # For example, 2D: (num_timesteps + 1, 1, 1, 1)
        #              1D: (num_timesteps + 1, 1)
        alpha_t = unsqueeze_to(alpha_t, len(self.input_shape) + 1)
        sigma_t = (1 - alpha_t ** 2).clamp(min=0) ** 0.5
        self.register_buffer("alpha_t", alpha_t)
        self.register_buffer("sigma_t", sigma_t)

    def loss(self, x: torch.Tensor):
        """
        Returns
        -------
        loss: (bsz, *input_shape)
        """
        bsz, *_ = x.shape
        t_sample = torch.randint(1, self.num_timesteps + 1, size=(bsz,), device=x.device)
        eps = torch.randn_like(x)
        x_t = self.alpha_t[t_sample] * x + self.sigma_t[t_sample] * eps
        pred_target = self.nn_module(x_t, t_sample)

        if self.target_type == "pred_x_0":
            gt_target = x
        elif self.target_type == "pred_eps":
            gt_target = eps
        elif self.target_type == "pred_v":
            gt_target = self.alpha_t[t_sample] * eps - self.sigma_t[t_sample] * x
        else:
            raise AssertionError(f"Invalid {self.target_type=}.")

        if self.loss_type == "l2":
            loss = 0.5 * (gt_target - pred_target) ** 2
        elif self.loss_type == "l1":
            loss = torch.abs(gt_target - pred_target)
        else:
            raise AssertionError(f"Invalid {self.loss_type=}.")

        return loss

    @torch.no_grad()
    def sample(self, bsz: int, device: str, num_sampling_timesteps: int | None = None):
        """
        Parameters
        ----------
        num_sampling_timesteps: int. If unspecified, defaults to self.num_timesteps.

        Returns
        -------
        samples: (num_sampling_timesteps + 1, bsz, *self.input_shape)
            index 0 corresponds to x_0
            index t corresponds to x_t
            last index corresponds to random noise
        """
        num_sampling_timesteps = num_sampling_timesteps or self.num_timesteps
        assert 1 <= num_sampling_timesteps <= self.num_timesteps

        x = torch.randn((bsz, *self.input_shape), device=device)
        t_start = torch.empty((bsz,), dtype=torch.int64, device=device)
        t_end = torch.empty((bsz,), dtype=torch.int64, device=device)

        subseq = torch.linspace(self.num_timesteps, 0, num_sampling_timesteps + 1).round()
        samples = torch.empty((num_sampling_timesteps + 1, bsz, *self.input_shape), device=device)
        samples[-1] = x

        # Note that t_start > t_end we're traversing pairwise down subseq.
        # For example, subseq here could be [500, 400, 300, 200, 100, 0]
        for idx, (scalar_t_start, scalar_t_end) in enumerate(zip(subseq[:-1], subseq[1:])):

            t_start.fill_(scalar_t_start)
            t_end.fill_(scalar_t_end)
            noise = torch.zeros_like(x) if scalar_t_end == 0 else torch.randn_like(x)

            if self.gamma_type == "ddim":
                gamma_t = 0.0
            elif self.gamma_type == "ddpm":
                gamma_t = (
                    self.sigma_t[t_end] / self.sigma_t[t_start] *
                    (1 - self.alpha_t[t_start] ** 2 / self.alpha_t[t_end] ** 2) ** 0.5
                )
            else:
                raise AssertionError(f"Invalid {self.gamma_type=}.")

            nn_out = self.nn_module(x, t_start)
            if self.target_type == "pred_x_0":
                pred_x_0 = nn_out
                pred_eps = (x - self.alpha_t[t_start] * nn_out) / self.sigma_t[t_start]
            elif self.target_type == "pred_eps":
                pred_x_0 = (x - self.sigma_t[t_start] * nn_out) / self.alpha_t[t_start]
                pred_eps = nn_out
            elif self.target_type == "pred_v":
                pred_x_0 = self.alpha_t[t_start] * x - self.sigma_t[t_start] * nn_out
                pred_eps = self.sigma_t[t_start] * x + self.alpha_t[t_start] * nn_out
            else:
                raise AssertionError(f"Invalid {self.target_type=}.")

            x = (
                (self.alpha_t[t_end] * pred_x_0) +
                (self.sigma_t[t_end] ** 2 - gamma_t ** 2).clamp(min=0) ** 0.5 * pred_eps +
                (gamma_t * noise)
            )
            samples[-1 - idx - 1] = x

        return samples


# Load data from https://www.openml.org/d/554
# (70000, 784) values between 0-255
x, _ = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, cache=True)

# Reshape to 32x32
x = rearrange(x, "b (h w) -> b h w", h=28, w=28)
x = np.pad(x, pad_width=((0, 0), (2, 2), (2, 2)))
x = rearrange(x, "b h w -> b (h w)")

# Standardize to [-1, 1]
input_mean = np.full((1, 32 ** 2), fill_value=127.5, dtype=np.float32)
input_sd = np.full((1, 32 ** 2), fill_value=127.5, dtype=np.float32)
x = ((x - input_mean) / input_sd).astype(np.float32)

# Define model
net = UNet(in_dim=1, embed_dim=128, dim_scales=(1, 2, 4, 8))
model = DiffusionModel(
    nn_module=net,
    input_shape=(1, 32, 32,),
    config=DiffusionModelConfig(
        num_timesteps=500,
        target_type="pred_x_0",
        gamma_type="ddim",
        noise_schedule_type="cosine",
    ),
)
model = model.to(DEVICE)


logger = logging.getLogger(__name__)

optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, ITERATIONS)

gname = "diff_gen" 
gen_path = os.path.join(MODEL_DIR, f"{gname}_{DATASET}_ep{ITERATIONS}_{DAY}.pth")
fname, ext = os.path.splitext(gen_path)

sample_dir = os.path.join(MODEL_DIR, f"{fname}")
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
    print(f'The new directory {sample_dir} has been created')

model.train()
print("Starting training Diffusion Model ...")
for i in range(ITERATIONS):
    x_batch = x[np.random.choice(len(x), BATCH_SIZE)]
    x_batch = torch.from_numpy(x_batch).to(DEVICE)
    x_batch = rearrange(x_batch, "b (h w) -> b () h w", h=32, w=32)

    start_t  = process_time()
    optimizer.zero_grad()
    loss = model.loss(x_batch).mean()
    loss.backward()
    optimizer.step()
    scheduler.step()

    elapsed_t = process_time() - start_t
    # if i % 100 == 0:
    #     logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}\t")
    

    if i % 20 == 0:
        print(f"[Iter-{i} (elapsed_time: {elapsed_t:.4f}secs)]: \t Loss: {loss.data:.4f}\t")
        print("Sampling and visualizing images ...")

        samples = model.sample(bsz=64, num_sampling_timesteps=None, device=DEVICE)

        timesteps = samples.shape[0]
        for t in range(timesteps):
            vutils.save_image(samples[t], f'{sample_dir}/gen_sample_t{t}.jpg', normalize=True)

    # end if
# end for
model.eval()




