import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
# For make_grid equivalent, we might still use torchvision or implement a JAX version
# If we want to stay pure JAX, we can implement a simple grid function.
from sklearn.manifold import TSNE
import seaborn as sns

import pandas as pd
import os

plt.rcParams["savefig.bbox"] = 'tight'

def set_grid(D, num_cells=1, is_random=False):
    """
    Args: 
        D (Array): (n, d1, d2) or (n, c, d1, d2) collection of image arrays
        
    Return:
    """
    # Simple JAX implementation of grid
    if len(D.shape) == 3:
        n, d1, d2 = D.shape
        D = D[:, jnp.newaxis, :, :]
    
    n, c, d1, d2 = D.shape
    
    grid_size = int(jnp.ceil(jnp.sqrt(num_cells)))
    grid = jnp.zeros((c, grid_size * d1, grid_size * d2))
    
    for i in range(num_cells):
        r = i // grid_size
        col = i % grid_size
        grid = grid.at[:, r*d1:(r+1)*d1, col*d2:(col+1)*d2].set(D[i])
        
    return grid

def show(imgs, cmap=plt.cm.gray):
    """
    Args:
        imgs (Array): image in the form of jnp.array (C, H, W)
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]
    
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        # img: (C, H, W) -> (H, W, C)
        img = jnp.transpose(img, (1, 2, 0))
        img_np = np.array(img)

        img_np = normalize(img_np, new_min=0, new_max=1)
        print("img_np :", np.min(img_np), np.max(img_np))
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze()
            
        axs[0, i].imshow(img_np,
            interpolation="nearest",
            cmap=cmap,
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
            np.array(vec).reshape(image_shape),
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


def plot_features_tsne(feat, labels, save_path):
    print('generating t-SNE plot...')
    tsne = TSNE(random_state=0)
    tsne_output = tsne.fit_transform(np.array(feat))

    df = pd.DataFrame(tsne_output, columns=['x', 'y'])
    df['targets'] = np.array(labels)

    plt.rcParams['figure.figsize'] = 10, 10
    sns.scatterplot(
        x='x', y='y',
        hue='targets',
        palette=sns.color_palette("hls", 10),
        data=df,
        marker='o',
        legend="full",
        alpha=0.5
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')

    plt.savefig(save_path, bbox_inches='tight')
    print('done!')
    plt.clf()
