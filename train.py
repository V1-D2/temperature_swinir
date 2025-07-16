# train.py
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
import gc

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model_adapter import TemperatureSwinIR
from data.data_loader import create_train_val_dataloaders
from data.temperature_utils import TemperatureProcessor
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim
from utils.logger import setup_logger
from utils.visualization import save_validation_images


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Setup directories
        self.setup_directories()

        # Setup logger
        self.logger = setup_logger(self.save_dir)
        self.logger.info(f"Training on device: {self.device}")

        # Setup temperature processor
        self.temp_processor = TemperatureProcessor(
            scale_factor=config['data']['scale_factor'],
            colormap=config['data']['colormap']
        )

        # Setup model
        self.setup_model()

        # Setup data loaders
        self.setup_dataloaders()

        # Setup training components
        self.setup_training()

        # Setup visualization
        self.writer = SummaryWriter(os.path.join(self.save_dir, 'tensorboard'))

        # Metrics tracking
        self.best_psnr = 0
        self.best_epoch = 0

    def setup_directories(self):
        """Create necessary directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.join(
            self.config['training']['save_dir'],
            f"{self.config['name']}_{timestamp}"
        )

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.save_dir, 'logs'), exist_ok=True)

        # Save config
        with open(os.path.join(self.save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f)

    def setup_model(self):
        """Initialize model"""
        self.logger.info("Setting up model...")

        self.model = TemperatureSwinIR(
            pretrained_path=self.config['model']['pretrained_path'],
            scale_factor=self.config['model']['scale_factor'],
            freeze_backbone=self.config['model']['freeze_backbone']
        ).to(self.device)

        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")

    def setup_dataloaders(self):
        """Setup data loaders"""
        self.logger.info("Setting up data loaders...")

        train_files = [
            os.path.join(self.config['data']['data_dir'], f"train_{i:02d}.npz")
            for i in range(self.config['data']['num_train_files'])
        ]

        val_file = os.path.join(self.config['data']['data_dir'], "val_00.npz")

        self.train_loader, self.val_loader = create_train_val_dataloaders(
            train_files=train_files,
            val_file=val_file,
            batch_size=self.config['training']['batch_size'],
            scale_factor=self.config['model']['scale_factor'],
            patch_size=self.config['data']['patch_size'],
            patch_height=self.config['data']['patch_height'],
            patch_width=self.config['data']['patch_width']
        )

        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")

    def setup_training(self):
        """Setup optimizer, scheduler, and loss"""
        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config['training']['lr'],
            betas=(0.9, 0.999),
            weight_decay=self.config['training']['weight_decay']
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['epochs'],
            eta_min=self.config['training']['lr_min']
        )

        # Loss function
        self.criterion = nn.L1Loss()

        # Load checkpoint if specified
        if self.config['training'].get('resume'):
            self.load_checkpoint(self.config['training']['resume'])

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()

        epoch_loss = 0
        epoch_psnr = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}/{self.config["training"]["epochs"]}')

        for i, batch in enumerate(pbar):
            # Move to device
            lq = batch['lq'].to(self.device)  # [B, 1, H, W]
            gt = batch['gt'].to(self.device)  # [B, 1, H, W]

            # Forward pass
            sr = self.model(lq)

            # Calculate loss
            loss = self.criterion(sr, gt)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config['training'].get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )

            self.optimizer.step()

            # Calculate PSNR
            with torch.no_grad():
                psnr = self.calculate_psnr(sr, gt)

            # Update metrics
            epoch_loss += loss.item()
            epoch_psnr += psnr

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{psnr:.2f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })

            # Log to tensorboard
            if i % self.config['training']['log_interval'] == 0:
                global_step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/psnr', psnr, global_step)

            # Clear cache periodically
            if i % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # Calculate epoch metrics
        epoch_loss /= len(self.train_loader)
        epoch_psnr /= len(self.train_loader)

        return epoch_loss, epoch_psnr

    def validate(self, epoch):
        """Validate model"""
        self.model.eval()

        val_loss = 0
        val_psnr = 0
        val_ssim = 0

        # For bicubic comparison
        bicubic_psnr = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc='Validation')):
                lq = batch['lq'].to(self.device)
                gt = batch['gt'].to(self.device)

                # Model prediction
                sr = self.model(lq)

                # Calculate loss
                loss = self.criterion(sr, gt)
                val_loss += loss.item()

                # Calculate metrics
                psnr = self.calculate_psnr(sr, gt)
                ssim = self.calculate_ssim(sr, gt)
                val_psnr += psnr
                val_ssim += ssim

                # Bicubic baseline
                bicubic = self.bicubic_upsample(lq, self.config['model']['scale_factor'])
                bicubic_psnr += self.calculate_psnr(bicubic, gt)

                # Save validation images
                if i < self.config['training']['num_val_images']:
                    self.save_validation_images(lq, sr, gt, epoch, i)

        # Calculate average metrics
        n_val = len(self.val_loader)
        val_loss /= n_val
        val_psnr /= n_val
        val_ssim /= n_val
        bicubic_psnr /= n_val

        # Log metrics
        self.writer.add_scalar('val/loss', val_loss, epoch)
        self.writer.add_scalar('val/psnr', val_psnr, epoch)
        self.writer.add_scalar('val/ssim', val_ssim, epoch)
        self.writer.add_scalar('val/bicubic_psnr', bicubic_psnr, epoch)
        self.writer.add_scalar('val/psnr_gain', val_psnr - bicubic_psnr, epoch)

        return val_loss, val_psnr, val_ssim, bicubic_psnr

    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")

        for epoch in range(1, self.config['training']['epochs'] + 1):
            # Train
            train_loss, train_psnr = self.train_epoch(epoch)

            # Validate
            val_loss, val_psnr, val_ssim, bicubic_psnr = self.validate(epoch)

            # Update scheduler
            self.scheduler.step()

            # Log epoch results
            self.logger.info(
                f"Epoch {epoch}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train PSNR: {train_psnr:.2f}, "
                f"Val Loss: {val_loss:.4f}, Val PSNR: {val_psnr:.2f}, "
                f"Val SSIM: {val_ssim:.4f}, Bicubic PSNR: {bicubic_psnr:.2f}, "
                f"PSNR Gain: {val_psnr - bicubic_psnr:.2f}"
            )

            # Save checkpoint
            if val_psnr > self.best_psnr:
                self.best_psnr = val_psnr
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_psnr, is_best=True)

            # Regular checkpoint
            if epoch % self.config['training']['save_interval'] == 0:
                self.save_checkpoint(epoch, val_psnr, is_best=False)

        self.logger.info(f"Training completed! Best PSNR: {self.best_psnr:.2f} at epoch {self.best_epoch}")

    def calculate_psnr(self, sr, gt):
        """Calculate PSNR for temperature data"""
        # Convert to numpy
        sr_np = sr.detach().cpu().numpy()
        gt_np = gt.detach().cpu().numpy()

        # Calculate PSNR for each sample in batch
        psnr_sum = 0
        for i in range(sr_np.shape[0]):
            psnr = self.temp_processor.calculate_temp_psnr(
                sr_np[i, 0], gt_np[i, 0], data_range=1.0
            )
            psnr_sum += psnr

        return psnr_sum / sr_np.shape[0]

    def calculate_ssim(self, sr, gt):
        """Calculate SSIM for temperature data"""
        # Convert to numpy and scale to [0, 255]
        sr_np = (sr.detach().cpu().numpy() * 255).astype(np.uint8)
        gt_np = (gt.detach().cpu().numpy() * 255).astype(np.uint8)

        ssim_sum = 0
        for i in range(sr_np.shape[0]):
            ssim = calculate_ssim(
                sr_np[i, 0], gt_np[i, 0],
                crop_border=0, test_y_channel=False
            )
            ssim_sum += ssim

        return ssim_sum / sr_np.shape[0]

    def bicubic_upsample(self, lq, scale):
        """Bicubic upsampling baseline"""
        b, c, h, w = lq.shape
        return torch.nn.functional.interpolate(
            lq, size=(h * scale, w * scale),
            mode='bicubic', align_corners=False
        )

    def save_validation_images(self, lq, sr, gt, epoch, batch_idx):
        """Save validation images"""
        save_path = os.path.join(
            self.save_dir, 'images',
            f'epoch_{epoch:04d}_batch_{batch_idx:04d}.png'
        )

        # Convert to temperature values (assuming normalized [0,1])
        temp_lr = lq[0, 0].cpu().numpy()
        temp_sr = sr[0, 0].cpu().numpy()
        temp_hr = gt[0, 0].cpu().numpy()

        # Save comparison
        self.temp_processor.save_temperature_comparison(
            temp_lr, temp_sr, temp_hr, save_path,
            title=f'Epoch {epoch} - Batch {batch_idx}'
        )

    def save_checkpoint(self, epoch, psnr, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'psnr': psnr,
            'config': self.config
        }

        if is_best:
            save_path = os.path.join(self.save_dir, 'models', 'best_model.pth')
        else:
            save_path = os.path.join(self.save_dir, 'models', f'checkpoint_epoch_{epoch:04d}.pth')

        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved checkpoint: {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"Resumed from epoch {start_epoch}")

        return start_epoch


def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Create trainer
    trainer = Trainer(config)

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()