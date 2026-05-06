import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
import math


def _to_numpy_image(sample):
    """Convert any image type to H×W×3 float32 numpy array in [0, 1]."""
    if isinstance(sample, dict):
        img = sample.get("image", sample.get("masked_image", None))
        if img is None:
            raise ValueError("Dict sample has no 'image' or 'masked_image' key.")
    else:
        img = sample

    if isinstance(img, str):
        img = np.array(Image.open(img).convert("RGB")).astype(np.float32) / 255.0

    elif isinstance(img, Image.Image):
        img = np.array(img.convert("RGB")).astype(np.float32) / 255.0

    elif isinstance(img, torch.Tensor):
        img = img.detach().cpu()
        if img.ndim == 4:                                   # (B, C, H, W) → first item
            img = img[0]
        if img.ndim == 3 and img.shape[0] in (1, 3, 4):    # (C, H, W) → (H, W, C)
            img = img.permute(1, 2, 0)
        img = img.numpy().astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0
        if img.ndim == 2 or img.shape[-1] == 1:            # grayscale → RGB
            img = np.repeat(img if img.ndim == 3 else img[..., None], 3, axis=-1)

    elif isinstance(img, np.ndarray):
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] != img.shape[1]:
            img = img.transpose(1, 2, 0)                   # (C, H, W) → (H, W, C)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        if img.max() > 1.0:
            img = img / 255.0
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    return np.clip(img, 0.0, 1.0)


def preview_image(
    samples,
    titles=None,
    ncols=4,
    cell_size=4,
    suptitle=None,
):
    """
    Display one or multiple images in a grid.

    Args:
        samples:    A single image OR a list of images.
                    Each item can be: torch.Tensor (C,H,W or B,C,H,W),
                    np.ndarray (H,W,C or C,H,W), PIL.Image, file path str,
                    or a dataset sample dict with an 'image'/'masked_image' key.
        titles:     A single str OR a list of str (one per image). Optional.
        ncols:      Max columns in the grid (default 4).
        cell_size:  Width/height of each subplot cell in inches (default 4).
        suptitle:   Optional overall figure title shown above the grid.
    """
    # ── Normalise inputs to lists ──────────────────────────────────────────────
    if not isinstance(samples, (list, tuple)):
        samples = [samples]

    n = len(samples)

    if titles is None:
        titles = [""] * n
    elif isinstance(titles, str):
        titles = [titles] * n
    elif len(titles) < n:
        titles = list(titles) + [""] * (n - len(titles))

    # ── Compute grid shape ─────────────────────────────────────────────────────
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(cell_size * ncols, cell_size * nrows),
        squeeze=False,          # always returns 2D array of axes
    )

    for i, (sample, title) in enumerate(zip(samples, titles)):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        try:
            img = _to_numpy_image(sample)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, f"Error:\n{e}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="red")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

    # ── Hide unused axes ───────────────────────────────────────────────────────
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row][col].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.01)

    plt.tight_layout()
    plt.show()