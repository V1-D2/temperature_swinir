# models/degradation_bsrgan.py
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import ndimage
import random


class TemperatureDegradation:
    """BSRGAN-style degradation adapted for temperature data"""

    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

        # Degradation parameters (adapted for temperature data)
        self.blur_kernel_size = [7, 9, 11, 13, 15, 17, 19, 21]
        self.blur_sigma = [0.2, 2]
        self.downsample_range = [0.5, 1.5]
        self.noise_range = [0, 0.05]  # Lower noise for temperature data
        self.jpeg_range = [30, 95]

    def random_blur_kernel(self, kernel_size):
        """Generate random blur kernel"""
        kernel_type = random.choice(['iso', 'aniso'])

        if kernel_type == 'iso':
            # Isotropic Gaussian
            sigma = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
            kernel = self.isotropic_gaussian_kernel(kernel_size, sigma)
        else:
            # Anisotropic Gaussian
            sigma_x = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
            sigma_y = random.uniform(self.blur_sigma[0], self.blur_sigma[1])
            rotation = random.uniform(0, 180)
            kernel = self.anisotropic_gaussian_kernel(kernel_size, sigma_x, sigma_y, rotation)

        return kernel

    def isotropic_gaussian_kernel(self, size, sigma):
        """Generate isotropic Gaussian kernel"""
        kernel = cv2.getGaussianKernel(size, sigma)
        kernel = np.outer(kernel, kernel)
        return kernel / kernel.sum()

    def anisotropic_gaussian_kernel(self, size, sigma_x, sigma_y, rotation):
        """Generate anisotropic Gaussian kernel"""
        kernel = np.zeros((size, size))
        center = size // 2

        for i in range(size):
            for j in range(size):
                x = i - center
                y = j - center

                # Rotate coordinates
                theta = np.deg2rad(rotation)
                x_rot = x * np.cos(theta) - y * np.sin(theta)
                y_rot = x * np.sin(theta) + y * np.cos(theta)

                # Gaussian
                kernel[i, j] = np.exp(-(x_rot ** 2 / (2 * sigma_x ** 2) + y_rot ** 2 / (2 * sigma_y ** 2)))

        return kernel / kernel.sum()

    def add_noise(self, img, noise_level):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_level, img.shape)
        return np.clip(img + noise, 0, 1)

    def add_jpeg_compression(self, img, quality):
        """Add JPEG compression artifacts (simulated for single channel)"""
        # Quantize to simulate compression
        img_uint8 = (img * 255).astype(np.uint8)

        # Simulate JPEG by DCT-based compression
        h, w = img_uint8.shape

        # Pad to multiple of 8
        h_pad = (8 - h % 8) % 8
        w_pad = (8 - w % 8) % 8
        img_padded = np.pad(img_uint8, ((0, h_pad), (0, w_pad)), mode='edge')

        # Simple quality-based quantization
        quantization = 1 + (100 - quality) / 10
        img_compressed = np.round(img_padded / quantization) * quantization

        # Crop back
        img_compressed = img_compressed[:h, :w]

        return np.clip(img_compressed / 255.0, 0, 1)

    def degradation_bsrgan(self, img_gt, lq_patchsize=64):
        """Apply BSRGAN degradation pipeline"""
        h, w = img_gt.shape

        # Random crop for training
        if h > lq_patchsize * self.scale_factor:
            top = random.randint(0, h - lq_patchsize * self.scale_factor)
            left = random.randint(0, w - lq_patchsize * self.scale_factor)
            img_gt = img_gt[top:top + lq_patchsize * self.scale_factor,
                     left:left + lq_patchsize * self.scale_factor]

        # 1. Blur
        kernel_size = random.choice(self.blur_kernel_size)
        kernel = self.random_blur_kernel(kernel_size)
        img_lq = cv2.filter2D(img_gt, -1, kernel)

        # 2. Downsample
        scale = self.scale_factor * random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_LINEAR)

        # 3. Add noise
        if random.random() < 0.5:
            noise_level = random.uniform(self.noise_range[0], self.noise_range[1])
            img_lq = self.add_noise(img_lq, noise_level)

        # 4. JPEG compression
        if random.random() < 0.5:
            quality = random.randint(self.jpeg_range[0], self.jpeg_range[1])
            img_lq = self.add_jpeg_compression(img_lq, quality)

        # 5. Resize to exact scale factor
        h_lq, w_lq = lq_patchsize, lq_patchsize
        img_lq = cv2.resize(img_lq, (w_lq, h_lq), interpolation=cv2.INTER_LINEAR)

        # Ensure GT is exact scale
        img_gt = cv2.resize(img_gt, (w_lq * self.scale_factor, h_lq * self.scale_factor),
                            interpolation=cv2.INTER_LINEAR)

        return img_lq.astype(np.float32), img_gt.astype(np.float32)

    def degradation_bsrgan_rect(self, img_gt, lq_patchsize_h=128, lq_patchsize_w=128):
        """Apply BSRGAN degradation pipeline with rectangular patches"""
        h, w = img_gt.shape

        # Random crop for training
        if h > lq_patchsize_h * self.scale_factor:
            top = random.randint(0, h - lq_patchsize_h * self.scale_factor)
            img_gt = img_gt[top:top + lq_patchsize_h * self.scale_factor, :]

        if w > lq_patchsize_w * self.scale_factor:
            left = random.randint(0, w - lq_patchsize_w * self.scale_factor)
            img_gt = img_gt[:, left:left + lq_patchsize_w * self.scale_factor]

        # Get actual patch size
        h_patch, w_patch = img_gt.shape

        # 1. Blur
        kernel_size = random.choice(self.blur_kernel_size)
        kernel = self.random_blur_kernel(kernel_size)
        img_lq = cv2.filter2D(img_gt, -1, kernel)

        # 2. Downsample with random scale variation
        scale = self.scale_factor * random.uniform(self.downsample_range[0], self.downsample_range[1])
        img_lq = cv2.resize(img_lq, (int(w_patch / scale), int(h_patch / scale)), interpolation=cv2.INTER_LINEAR)

        # 3. Add noise
        if random.random() < 0.5:
            noise_level = random.uniform(self.noise_range[0], self.noise_range[1])
            img_lq = self.add_noise(img_lq, noise_level)

        # 4. JPEG compression
        if random.random() < 0.5:
            quality = random.randint(self.jpeg_range[0], self.jpeg_range[1])
            img_lq = self.add_jpeg_compression(img_lq, quality)

        # 5. Resize to exact scale factor
        img_lq = cv2.resize(img_lq, (lq_patchsize_w, lq_patchsize_h), interpolation=cv2.INTER_LINEAR)

        # Ensure GT is exact scale
        img_gt = cv2.resize(img_gt, (lq_patchsize_w * self.scale_factor, lq_patchsize_h * self.scale_factor),
                            interpolation=cv2.INTER_LINEAR)

        return img_lq.astype(np.float32), img_gt.astype(np.float32)