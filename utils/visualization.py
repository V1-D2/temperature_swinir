# utils/visualization.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional
import cv2


def save_validation_images(lq, sr, gt, save_path, temp_processor, metadata=None):
    """Save validation comparison images"""
    # Convert tensors to numpy arrays
    if isinstance(lq, torch.Tensor):
        lq = lq.detach().cpu().numpy()
    if isinstance(sr, torch.Tensor):
        sr = sr.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()

    # Remove batch and channel dimensions if present
    if lq.ndim == 4:
        lq = lq[0, 0]
    if sr.ndim == 4:
        sr = sr[0, 0]
    if gt.ndim == 4:
        gt = gt[0, 0]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Temperature maps
    vmin = min(lq.min(), sr.min(), gt.min())
    vmax = max(lq.max(), sr.max(), gt.max())

    # Row 1: Temperature maps
    im1 = axes[0, 0].imshow(lq, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'LR ({lq.shape[0]}x{lq.shape[1]})')
    axes[0, 0].axis('off')

    im2 = axes[0, 1].imshow(sr, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'SR ({sr.shape[0]}x{sr.shape[1]})')
    axes[0, 1].axis('off')

    im3 = axes[0, 2].imshow(gt, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title(f'HR ({gt.shape[0]}x{gt.shape[1]})')
    axes[0, 2].axis('off')

    # Row 2: Error maps and histograms
    error_sr = np.abs(sr - gt)
    error_bicubic = np.abs(cv2.resize(lq, (gt.shape[1], gt.shape[0]),
                                      interpolation=cv2.INTER_CUBIC) - gt)

    im4 = axes[1, 0].imshow(error_sr, cmap='hot')
    axes[1, 0].set_title(f'SR Error (MAE: {error_sr.mean():.4f})')
    axes[1, 0].axis('off')

    im5 = axes[1, 1].imshow(error_bicubic, cmap='hot')
    axes[1, 1].set_title(f'Bicubic Error (MAE: {error_bicubic.mean():.4f})')
    axes[1, 1].axis('off')

    # Histogram
    axes[1, 2].hist(sr.flatten(), bins=50, alpha=0.5, label='SR', density=True)
    axes[1, 2].hist(gt.flatten(), bins=50, alpha=0.5, label='HR', density=True)
    axes[1, 2].set_xlabel('Temperature (normalized)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].legend()
    axes[1, 2].set_title('Temperature Distribution')

    # Add colorbars
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04)
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Add metadata if provided
    if metadata:
        fig.suptitle(metadata, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(log_file, save_path):
    """Plot training curves from log file"""
    # Parse log file
    epochs = []
    train_loss = []
    train_psnr = []
    val_loss = []
    val_psnr = []
    val_ssim = []

    with open(log_file, 'r') as f:
        for line in f:
            if 'Epoch' in line and 'Train Loss' in line:
                parts = line.split()
                epoch = int(parts[2].split('/')[0])
                epochs.append(epoch)

                # Extract metrics
                for i, part in enumerate(parts):
                    if part == 'Loss:':
                        if 'Train' in parts[i - 1]:
                            train_loss.append(float(parts[i + 1].rstrip(',')))
                        elif 'Val' in parts[i - 1]:
                            val_loss.append(float(parts[i + 1].rstrip(',')))
                    elif part == 'PSNR:':
                        if i > 0 and 'Train' in ' '.join(parts[:i]):
                            train_psnr.append(float(parts[i + 1].rstrip(',')))
                        else:
                            val_psnr.append(float(parts[i + 1].rstrip(',')))
                    elif part == 'SSIM:':
                        val_ssim.append(float(parts[i + 1].rstrip(',')))

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Loss plot
    axes[0, 0].plot(epochs, train_loss, label='Train')
    axes[0, 0].plot(epochs, val_loss, label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # PSNR plot
    axes[0, 1].plot(epochs, train_psnr, label='Train')
    axes[0, 1].plot(epochs, val_psnr, label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('PSNR')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # SSIM plot
    axes[1, 0].plot(epochs, val_ssim)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('Validation SSIM')
    axes[1, 0].grid(True)

    # Learning rate plot (if available)
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


 # Add this import at the top