# data/temperature_utils.py
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Tuple, Optional, Union


class TemperatureProcessor:
    """Utilities for processing temperature data"""

    def __init__(self, scale_factor: float = 0.001, colormap: str = 'turbo'):
        self.scale_factor = scale_factor
        self.colormap = colormap
        self.cmap = cm.get_cmap(colormap)

    def kelvin_to_celsius(self, temp_kelvin: np.ndarray) -> np.ndarray:
        """Convert temperature from Kelvin to Celsius"""
        return temp_kelvin * self.scale_factor - 273.15

    def celsius_to_kelvin(self, temp_celsius: np.ndarray) -> np.ndarray:
        """Convert temperature from Celsius to Kelvin"""
        return (temp_celsius + 273.15) / self.scale_factor

    def filter_outliers(self, temp: np.ndarray,
                        lower_percentile: float = 1.0,
                        upper_percentile: float = 99.0) -> Tuple[np.ndarray, dict]:
        """Filter temperature outliers using percentiles"""
        # Calculate percentiles excluding NaN
        lower = np.nanpercentile(temp, lower_percentile)
        upper = np.nanpercentile(temp, upper_percentile)

        # Clip values
        temp_filtered = np.clip(temp, lower, upper)

        # Store statistics
        stats = {
            'original_min': np.nanmin(temp),
            'original_max': np.nanmax(temp),
            'filtered_min': lower,
            'filtered_max': upper,
            'n_clipped': np.sum((temp < lower) | (temp > upper))
        }

        return temp_filtered, stats

    def normalize_temperature(self, temp: np.ndarray,
                              temp_min: Optional[float] = None,
                              temp_max: Optional[float] = None) -> Tuple[np.ndarray, dict]:
        """Normalize temperature to [0, 1] range"""
        if temp_min is None:
            temp_min = np.nanmin(temp)
        if temp_max is None:
            temp_max = np.nanmax(temp)

        # Avoid division by zero
        if temp_max - temp_min < 1e-6:
            return np.zeros_like(temp), {'min': temp_min, 'max': temp_max}

        temp_norm = (temp - temp_min) / (temp_max - temp_min)
        temp_norm = np.clip(temp_norm, 0, 1)

        return temp_norm, {'min': temp_min, 'max': temp_max}

    def denormalize_temperature(self, temp_norm: np.ndarray,
                                temp_min: float,
                                temp_max: float) -> np.ndarray:
        """Denormalize temperature from [0, 1] to original range"""
        return temp_norm * (temp_max - temp_min) + temp_min

    def temperature_to_rgb(self, temp_norm: np.ndarray) -> np.ndarray:
        """Convert normalized temperature [0,1] to RGB using colormap"""
        # Ensure 2D array
        if len(temp_norm.shape) == 3:
            temp_norm = temp_norm.squeeze()

        # Apply colormap
        rgb = self.cmap(temp_norm)[:, :, :3]  # Remove alpha channel
        return (rgb * 255).astype(np.uint8)

    def rgb_to_temperature(self, rgb: np.ndarray,
                           temp_min: float,
                           temp_max: float) -> np.ndarray:
        """Convert RGB back to temperature (approximate)"""
        # Convert to grayscale as approximation
        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb

        # Normalize to [0, 1]
        gray_norm = gray.astype(np.float32) / 255.0

        # Denormalize to temperature range
        temp = self.denormalize_temperature(gray_norm, temp_min, temp_max)

        return temp

    def calculate_temp_psnr(self, temp1: np.ndarray,
                            temp2: np.ndarray,
                            data_range: Optional[float] = None) -> float:
        """Calculate PSNR between two temperature arrays"""
        if data_range is None:
            data_range = max(temp1.max() - temp1.min(),
                             temp2.max() - temp2.min())

        mse = np.mean((temp1 - temp2) ** 2)
        if mse == 0:
            return float('inf')

        psnr = 20 * np.log10(data_range / np.sqrt(mse))
        return psnr

    def save_temperature_comparison(self,
                                    temp_lr: np.ndarray,
                                    temp_sr: np.ndarray,
                                    temp_hr: np.ndarray,
                                    save_path: str,
                                    title: str = "Temperature Comparison"):
        """Save comparison visualization of LR, SR, and HR temperatures"""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Temperature maps
        vmin = min(temp_lr.min(), temp_sr.min(), temp_hr.min())
        vmax = max(temp_lr.max(), temp_sr.max(), temp_hr.max())

        im1 = axes[0].imshow(temp_lr, cmap=self.colormap, vmin=vmin, vmax=vmax)
        axes[0].set_title('LR (Input)')
        axes[0].axis('off')

        im2 = axes[1].imshow(temp_sr, cmap=self.colormap, vmin=vmin, vmax=vmax)
        axes[1].set_title('SR (Output)')
        axes[1].axis('off')

        im3 = axes[2].imshow(temp_hr, cmap=self.colormap, vmin=vmin, vmax=vmax)
        axes[2].set_title('HR (Target)')
        axes[2].axis('off')

        # Error map
        error = np.abs(temp_sr - temp_hr)
        im4 = axes[3].imshow(error, cmap='hot')
        axes[3].set_title('Absolute Error')
        axes[3].axis('off')

        # Add colorbars
        plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
        plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def tensor_to_temperature(tensor: torch.Tensor,
                          temp_min: float,
                          temp_max: float) -> np.ndarray:
    """Convert normalized tensor to temperature array"""
    # tensor: [B, 1, H, W] or [1, H, W] or [H, W]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    # Convert to numpy
    temp_norm = tensor.cpu().numpy()

    # Denormalize
    temp = temp_norm * (temp_max - temp_min) + temp_min

    return temp