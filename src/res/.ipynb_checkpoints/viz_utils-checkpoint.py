import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms.functional as F
from torchvision.utils import make_grid

plt.rcParams["savefig.bbox"] = 'tight'

def set_grid(D, num_cells=1, is_random=False):
    """
    Args: 
        D (Tensor): (n, c, d1, d2) collection of image tensors
        
    Return:
    """
    if type(D) is np.ndarray:
        A = torch.from_numpy(D)
    else:
        A = D
        
    if len(A.shape) == 3:
        (n, d1, d2) = A.shape
        c = 1
    elif len(A.shape) == 4:
        (n, c, d1, d2) = A.shape
    
    if not is_random:
        img_list = [torch.reshape(A[i], (c, d1, d2)) for i in range(num_cells)]
    else:
        img_list = [torch.reshape(A[i], (c, d1, d2)) for i in range(num_cells)]
        
    return make_grid(img_list)

def show(imgs, cmap=plt.cm.gray):
    """
    Args:
        imgs (list): list of images in the form of torch.tensor
    """
    if not isinstance(imgs, list):
        imgs = [imgs]
    
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = torch.permute(img, (1, 2, 0))
        img_np = img.numpy()

        img_np = normalize(img_np, new_min=0, new_max=1)
        print("img_np :", np.min(img_np), np.max(img_np))
        axs[0, i].imshow(img_np,
            interpolation="nearest",
            cmap=cmap,
            # vmin=-vmax,
            # vmax=vmax
        )
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        

def show_components(images, image_shape, n_row=2, n_col=3, cmap=plt.cm.gray):
    fig, axs = plt.subplots(
        nrows=n_row,
        ncols=n_col,
        figsize=(2.0 * n_col, 2.3 * n_row),
        facecolor="white",
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0, wspace=0)
    fig.set_edgecolor("black")
    for ax, vec in zip(axs.flat, images):
        vmax = max(vec.max(), -vec.min())
        im = ax.imshow(
            vec.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axis("off")

    fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
    plt.show()
    
    
def normalize(x, new_min=0, new_max=255):
    old_min = np.min(x)
    old_max = np.max(x)
    xn = (x - old_min) * ((new_max - new_min) / (old_max - old_min)) + new_min
    return xn

