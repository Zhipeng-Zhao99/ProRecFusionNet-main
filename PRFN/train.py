"""
Training script for  image‑processing models based on ProRecFusionNet.
"""

import logging
import shutil
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from pytorch_msssim import ms_ssim
from tqdm import tqdm
import os, random, numpy as np, torch
from model.PRFN import ProRecFusionNet
from data.data import load_training_dataset, load_validation_dataset
from args import config


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# -----------------------------------------------------------------------------
# Metric calculation functions
# -----------------------------------------------------------------------------
def calculate_psnr(img1, img2, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio."""
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def calculate_ssim(img1, img2, window_size=11, reduction='mean', max_val=1.0):
    """Calculate Structural Similarity Index."""
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)

    sigma1 = np.sqrt(np.mean((img1 - mu1) ** 2))
    sigma2 = np.sqrt(np.mean((img2 - mu2) ** 2))
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))

    return ssim


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------
class Trainer:
    """Trainer for the ProRecFusionNet edge‑enhanced fusion model."""

    def __init__(self):
        # ──────────────── Hyper‑parameters ────────────────
        self.num_epochs = config.epochs
        self.batch_size = config.batch_size
        self.learning_rate = config.lr

        # ──────────────── Model & optimiser ────────────────
        self.model = ProRecFusionNet()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=config.lr_decay_freq, gamma=0.1)

        # ──────────────── Datasets ────────────────
        self.train_dataset = load_training_dataset(config.train_data_dir)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
        )
        self.eval_dataset = load_validation_dataset(config.eval_data_dir)
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=6,
        )

        # ──────────────── Losses ────────────────
        self.mse_loss = nn.MSELoss()
        # Store MS‑SSIM function as a field for convenience
        self.ms_ssim = ms_ssim

        # ──────────────── Device ────────────────
        self.device = torch.device("cuda" if config.cuda and torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.mse_loss.to(self.device)
        logger.info(f"Using device: {self.device}")

        # ──────────────── Book‑keeping ────────────────
        self.records = {"epoch": [], "loss": [], "psnr": [], "ssim": []}
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_backup_dir = config.checkpoint_backup_dir
        self.log_dir = config.log_dir
        self.record_dir = config.record_dir
        self.current_epoch = 1
        self.best_psnr = 0.0

        self._create_directories()
        self._init_record_files()

    # ---------------------------------------------------------------------
    # House‑keeping helpers
    # ---------------------------------------------------------------------
    def _create_directories(self):
        for directory in [self.log_dir, self.record_dir, self.checkpoint_dir, self.checkpoint_backup_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created or verified directory: {directory}")

    def _init_record_files(self):
        self.train_loss_file = self.record_dir / "train_loss_record.txt"
        self.epoch_time_file = self.record_dir / "epoch_time_record.txt"
        self.eval_loss_file = self.record_dir / "eval_loss_record.txt"

        logger.info("Log files will be appended. No old logs will be deleted.")
        for f in [self.train_loss_file, self.epoch_time_file, self.eval_loss_file]:
            f.touch(exist_ok=True)

    # ---------------------------------------------------------------------
    # Check‑pointing helpers
    # ---------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "records": self.records,
        }
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            shutil.copy(latest_path, best_path)
            logger.info(f"✓ New best model saved (PSNR: {self.best_psnr:.2f})")

        if epoch % config.checkpoint_save_interval == 0:
            backup_path = self.checkpoint_backup_dir / f"epoch_{epoch}.pth"
            shutil.copy(latest_path, backup_path)
            logger.info(f"✓ Backed up model at epoch {epoch}")

    # ---------------------------------------------------------------------
    # Loss calculation helper
    # ---------------------------------------------------------------------
    def calculate_loss(self, outputs, targets):
        """Calculate the compound MSE + 0.1*(1-MS_SSIM) loss."""
        ssim_score = self.ms_ssim(outputs, targets, data_range=1.0, size_average=True)
        loss = self.mse_loss(outputs, targets) + 0.1 * (1.0 - ssim_score)
        return loss, ssim_score

    # ---------------------------------------------------------------------
    # Training & validation loops
    # ---------------------------------------------------------------------
    def train(self):
        """Execute the training loop with periodic validation."""
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        logger.info(f"Training size: {len(self.train_dataset)}, Validation size: {len(self.eval_dataset)}")

        steps_per_epoch = len(self.train_loader)
        logger.info(f"Steps per epoch: {steps_per_epoch}")

        self.model.train()
        global_step = (self.current_epoch - 1) * steps_per_epoch

        for epoch in range(self.current_epoch, self.num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(f"Epoch {epoch}/{self.num_epochs}, LR: {current_lr:.6e}")
            running_loss = 0.0

            with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch}/{self.num_epochs}") as pbar:
                for pan, lr, lr_u, ms in self.train_loader:
                    global_step += 1
                    pan, lr, ms = pan.to(self.device), lr.to(self.device), ms.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(pan, lr)

                    # Calculate loss using helper function
                    loss, ssim_score = self.calculate_loss(outputs, ms)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    running_loss += loss.item()
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "ssim": f"{ssim_score.item():.4f}"})
                    pbar.update(1)

            avg_loss = running_loss / steps_per_epoch
            self.records["epoch"].append(epoch)
            self.records["loss"].append(avg_loss)

            # Record training loss to file
            with open(self.train_loss_file, "a") as f:
                f.write(f"Epoch {epoch}: Loss {avg_loss:.6f}\n")

            logger.info(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.6f}")

            # Save checkpoint at each epoch
            self._save_checkpoint(epoch)

            # Record epoch time
            epoch_minutes = (time.time() - epoch_start) / 60.0
            with open(self.epoch_time_file, "a") as f:
                f.write(f"Epoch {epoch} training time: {epoch_minutes:.3f} minutes\n")

            # Periodic validation
            if epoch % config.validation_interval == 0:
                validation_start = time.time()
                self.validate()
                validation_minutes = (time.time() - validation_start) / 60.0
                logger.info(f"Validation time: {validation_minutes:.2f} minutes")

            # Step the learning rate scheduler
            self.scheduler.step()

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------
    def validate(self):
        """Evaluate model performance on validation dataset."""
        self.model.eval()
        psnr_vals, ssim_vals, loss_vals = [], [], []

        logger.info("Running validation...")

        # Initialize validation data file for this epoch
        with open(self.eval_loss_file, "a") as f:
            f.write(f"\n{'-'*50}\n")
            f.write(f"Validation at Epoch {self.current_epoch}\n")
            f.write(f"{'-'*50}\n")

        # Process validation data with progress bar
        with torch.no_grad():
            val_loop = tqdm(self.eval_loader, desc="Validating", leave=False)
            for batch_idx, (pan, lr, lr_u, target) in enumerate(val_loop):
                pan, lr, target = pan.to(self.device), lr.to(self.device), target.to(self.device)
                fused = self.model(pan, lr)

                # Calculate the same loss as in training
                loss, ssim_score = self.calculate_loss(fused, target)

                # Also calculate PSNR for model selection
                psnr = calculate_psnr(fused, target, 1.0)

                # Record values
                loss_vals.append(loss.item())
                psnr_vals.append(psnr if not torch.is_tensor(psnr) else psnr.item())
                ssim_vals.append(ssim_score.item())

                # Update progress bar
                val_loop.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "PSNR": f"{psnr:.2f}",
                    "SSIM": f"{ssim_score.item():.4f}"
                })

                # Log detailed metrics to file only (not to console)
                with open(self.eval_loss_file, "a") as f:
                    f.write(
                        f"Batch {batch_idx+1}: Loss: {loss.item():.6f}, PSNR: {psnr:.4f}, SSIM: {ssim_score.item():.4f}\n"
                    )

        # Calculate average metrics
        avg_loss = float(np.mean(loss_vals))
        avg_psnr = float(np.mean(psnr_vals))
        avg_ssim = float(np.mean(ssim_vals))

        # Record metrics for this epoch
        self.records["psnr"].append(avg_psnr)
        self.records["ssim"].append(avg_ssim)

        # Log summary to file
        with open(self.eval_loss_file, "a") as f:
            f.write(f"{'-'*50}\n")
            f.write(f"Avg Loss: {avg_loss:.6f}, Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.6f}\n")

            # Handle previous best PSNR comparison
            if len(self.records["psnr"]) > 1:
                prev_best = max(self.records["psnr"][:-1])  # All records except current
                f.write(f"Previous best PSNR: {prev_best:.4f}, Current PSNR: {avg_psnr:.4f}, Diff: {avg_psnr-prev_best:.4f}\n")
            f.write(f"{'-'*50}\n\n")

        # Check if this is a new best model (fix the empty sequence error)
        is_best = False
        if len(self.records["psnr"]) <= 1:  # First validation or only current
            is_best = True
        else:
            prev_best = max(self.records["psnr"][:-1])  # All records except current
            is_best = avg_psnr > prev_best

        if is_best:
            self.best_psnr = avg_psnr

        # Log summary to console
        logger.info(f"Validation results - Avg Loss: {avg_loss:.4f}, Avg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.4f}")

        # If we have previous records, show improvement
        if len(self.records["psnr"]) > 1:
            prev_best = max(self.records["psnr"][:-1])
            change = avg_psnr - prev_best
            if change > 0:
                logger.info(f"PSNR improved by {change:.2f} dB")
            else:
                logger.info(f"PSNR decreased by {abs(change):.2f} dB from best ({prev_best:.2f})")

        # Save best model if needed
        if is_best:
            self._save_checkpoint(self.current_epoch, is_best=True)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def main():
    set_seed(config.seed)
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()