# models/model_adapter.py
import torch
import torch.nn as nn
import numpy as np
from models.network_swinir import SwinIR
import matplotlib.cm as cm
import cv2


class TemperatureSwinIR(nn.Module):
    """Adapter for SwinIR to handle single-channel temperature data"""

    def __init__(self, pretrained_path, scale_factor=2, freeze_backbone=False):
        super().__init__()
        self.scale_factor = scale_factor

        # Load pretrained SwinIR model
        self.swinir = SwinIR(
            upscale=scale_factor,
            in_chans=3,
            img_size=48,  # Match pretrained model
            window_size=8,
            img_range=1.,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler='pixelshuffle',
            resi_connection='1conv'
        )

        # Load pretrained weights
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'params' in checkpoint:
            self.swinir.load_state_dict(checkpoint['params'], strict=True)
        else:
            self.swinir.load_state_dict(checkpoint, strict=True)

        # Input adapter: 1 channel -> 3 channels
        self.input_adapter = nn.Conv2d(1, 3, 1, 1, 0)
        nn.init.xavier_uniform_(self.input_adapter.weight)

        # Output adapter: 3 channels -> 1 channel
        self.output_adapter = nn.Conv2d(3, 1, 1, 1, 0)
        nn.init.xavier_uniform_(self.output_adapter.weight)

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.swinir.parameters():
                param.requires_grad = False

    def forward(self, x):
        # x: [B, 1, H, W] temperature data normalized to [0, 1]

        # Adapt to 3 channels
        x_rgb = self.input_adapter(x)

        # Apply SwinIR
        x_sr = self.swinir(x_rgb)

        # Convert back to single channel
        x_out = self.output_adapter(x_sr)

        return x_out

    def temperature_to_rgb(self, temp_tensor, colormap='turbo'):
        """Convert temperature tensor to RGB for visualization"""
        # temp_tensor: [B, 1, H, W]
        batch_size = temp_tensor.shape[0]
        rgb_batch = []

        for i in range(batch_size):
            temp = temp_tensor[i, 0].cpu().numpy()

            # Apply colormap
            if colormap == 'turbo':
                cmap = cm.get_cmap('turbo')
            else:
                cmap = cm.get_cmap(colormap)

            # Normalize to [0, 1] if not already
            if temp.max() > 1.0:
                temp = (temp - temp.min()) / (temp.max() - temp.min())

            rgb = cmap(temp)[:, :, :3]  # Remove alpha channel
            rgb = torch.from_numpy(rgb).permute(2, 0, 1).float()
            rgb_batch.append(rgb)

        return torch.stack(rgb_batch).to(temp_tensor.device)