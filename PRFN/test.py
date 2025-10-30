"""
Testing script for image-processing models based on ProRecFusionNet.
"""

import logging
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from osgeo import gdal, osr
from model.PRFN import ProRecFusionNet
from data.data import load_testing_dataset
from data.dataset import denormalize_image
from args import config

# Configure logger
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Custom utility functions
# -----------------------------------------------------------------------------

def cc(img1, img2):
    """Calculate Cross Correlation between two images."""
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()

    # Handle multi-channel images
    if img1.ndim == 3:  # Multi-channel image
        # Process each channel separately and average
        cc_sum = 0
        for i in range(img1.shape[0]):
            img1_band = img1[i].flatten() - np.mean(img1[i])
            img2_band = img2[i].flatten() - np.mean(img2[i])

            numerator = np.sum(img1_band * img2_band)
            denominator = np.sqrt(np.sum(img1_band ** 2) * np.sum(img2_band ** 2))

            if denominator > 1e-10:  # Avoid division by zero
                cc_sum += numerator / denominator

        return cc_sum / img1.shape[0]
    else:  # Single-channel image
        img1_flat = img1.flatten() - np.mean(img1)
        img2_flat = img2.flatten() - np.mean(img2)

        numerator = np.sum(img1_flat * img2_flat)
        denominator = np.sqrt(np.sum(img1_flat ** 2) * np.sum(img2_flat ** 2))

        if denominator > 1e-10:  # Avoid division by zero
            return numerator / denominator
        else:
            return 0.0


def save_geotiff(output_path, raster_origin, pixel_width, pixel_height, array, band_size):
    """
    Save a numpy array as a GeoTIFF raster file.
    """
    # Extract origin coordinates
    origin_x = raster_origin[0]
    origin_y = raster_origin[1]

    # Get array dimensions based on band size
    if band_size == 4:
        # Multi-band image: array shape is [bands, rows, cols]
        rows = array.shape[1]
        cols = array.shape[2]
    elif band_size == 1:
        # Single-band image: array shape is [rows, cols]
        rows = array.shape[0]
        cols = array.shape[1]
    else:
        raise ValueError(f"Unsupported band size: {band_size}. Must be 1 or 4.")

    # Create driver
    driver = gdal.GetDriverByName('GTiff')

    # Create output raster
    out_raster = driver.Create(output_path, cols, rows, band_size, gdal.GDT_UInt16)

    # Set geotransform
    out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0, pixel_height))

    # Write data based on band size
    if band_size == 4:
        for i in range(1, 5):
            band = out_raster.GetRasterBand(i)
            band.WriteArray(array[i - 1, :, :])
    elif band_size == 1:
        band = out_raster.GetRasterBand(1)
        band.WriteArray(array)

    # Set projection
    out_raster_srs = osr.SpatialReference()
    out_raster_srs.ImportFromEPSG(4326)  # WGS84
    out_raster.SetProjection(out_raster_srs.ExportToWkt())

    # Flush cache
    if band_size == 1:
        band.FlushCache()
    else:
        out_raster.FlushCache()

    # Close dataset
    out_raster = None


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


def calculate_sam(img1, img2):
    """
    Calculate Spectral Angle Mapper (SAM) between two images.
    Optimized implementation for remote sensing imagery.
    """
    # Convert to numpy if needed
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()

    # Debug image dimensions
    logger.debug(f"SAM calculation - Image shapes: img1={img1.shape}, img2={img2.shape}")

    # Handle batch dimension if present
    if img1.ndim == 4:  # Batch of images
        img1 = img1[0]  # Take first image
        img2 = img2[0]

    # Ensure we're working with [C, H, W] format for multispectral images
    if img1.ndim == 3 and img1.shape[0] >= 3:  # [C, H, W] with C bands
        # Reshape to [bands, pixels]
        bands1 = img1.reshape(img1.shape[0], -1)  # [C, H*W]
        bands2 = img2.reshape(img2.shape[0], -1)  # [C, H*W]

        # Each column is a pixel with C spectral bands
        # Compute SAM for each pixel
        sam_values = []

        # Process in batches to improve speed for large images
        batch_size = 10000
        num_pixels = bands1.shape[1]

        for i in range(0, num_pixels, batch_size):
            end_idx = min(i + batch_size, num_pixels)

            # Extract batch of pixels
            b1 = bands1[:, i:end_idx]  # [C, batch]
            b2 = bands2[:, i:end_idx]  # [C, batch]

            # Compute dot products
            # sum(a_i * b_i) for each pixel
            dots = np.sum(b1 * b2, axis=0)  # [batch]

            # Compute magnitudes
            # sqrt(sum(a_i^2)) for each pixel
            norm1 = np.sqrt(np.sum(b1 * b1, axis=0))  # [batch]
            norm2 = np.sqrt(np.sum(b2 * b2, axis=0))  # [batch]

            # Find valid pixels (non-zero magnitudes)
            valid = (norm1 > 1e-6) & (norm2 > 1e-6)

            if np.any(valid):
                # Compute cos(angle)
                cos_angles = dots[valid] / (norm1[valid] * norm2[valid])

                # Handle numerical issues
                cos_angles = np.clip(cos_angles, -1.0, 1.0)

                # Convert to angles in degrees
                angles = np.degrees(np.arccos(cos_angles))

                sam_values.extend(angles)

        if not sam_values:
            logger.warning("No valid pixels for SAM calculation")
            return 0.0

        # Return mean SAM value
        mean_sam = np.mean(sam_values)
        logger.debug(f"Mean SAM value: {mean_sam:.4f} degrees")
        return mean_sam

    elif img1.ndim == 3 and img1.shape[2] >= 3:  # [H, W, C] format
        # Reshape to [pixels, bands]
        pixels1 = img1.reshape(-1, img1.shape[2])  # [H*W, C]
        pixels2 = img2.reshape(-1, img2.shape[2])  # [H*W, C]

        # Calculate vector magnitudes for each pixel
        magnitudes1 = np.sqrt(np.sum(pixels1 ** 2, axis=1))
        magnitudes2 = np.sqrt(np.sum(pixels2 ** 2, axis=1))

        # Calculate dot product for each pixel
        dot_products = np.sum(pixels1 * pixels2, axis=1)

        # Find valid pixels (non-zero magnitudes)
        valid_pixels = (magnitudes1 > 1e-6) & (magnitudes2 > 1e-6)

        if np.sum(valid_pixels) == 0:
            logger.warning("No valid pixels found for SAM calculation")
            return 0.0

        # Calculate cosine of angles
        cos_angles = dot_products[valid_pixels] / (magnitudes1[valid_pixels] * magnitudes2[valid_pixels])

        # Clip values to valid range [-1, 1] to prevent numerical errors
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        # Calculate angles in radians, convert to degrees
        angles_deg = np.degrees(np.arccos(cos_angles))

        # Return mean angle
        mean_angle = np.mean(angles_deg)
        logger.debug(f"Calculated SAM: {mean_angle:.4f} degrees")
        return mean_angle

    else:
        logger.warning(f"Unsupported image format for SAM calculation: {img1.shape}")
        logger.warning("SAM requires multi-spectral images with at least 3 bands")
        return 0.0


def calculate_ergas(img1, img2, ratio=4):
    """
    Calculate ERGAS (Erreur Relative Globale Adimensionnelle de Synthèse).
    ERGAS = 100 * d/l * sqrt(1/N * sum((RMSE(i)/MEAN(i))^2))
    """
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()

    if img1.ndim == 3:
        bands = img1.shape[0]
        rmse_sum = 0

        for i in range(bands):
            # Calculate RMSE for current band
            rmse = np.sqrt(np.mean((img1[i] - img2[i]) ** 2))
            # Calculate mean of reference image for current band
            mean = np.abs(np.mean(img2[i]))

            # Avoid division by zero
            if mean < 1e-8:
                continue

            # Accumulate (RMSE/MEAN)^2
            rmse_sum += (rmse / mean) ** 2

        # Ensure at least one valid band
        if bands > 0:
            # Calculate ERGAS
            ergas = 100.0 * (1.0 / ratio) * np.sqrt(rmse_sum / bands)
        else:
            ergas = 0
    else:
        # Single band case
        rmse = np.sqrt(np.mean((img1 - img2) ** 2))
        mean = np.abs(np.mean(img2))

        if mean < 1e-8:
            ergas = 0
        else:
            ergas = 100.0 * (1.0 / ratio) * (rmse / mean)

    return ergas


def compute_metrics(img1, img2):
    """Compute various image quality metrics."""
    # Calculate all metrics
    psnr = calculate_psnr(img1, img2, 1.0)
    ssim = calculate_ssim(img1, img2, 11, "mean", 1.0)
    cc_val = cc(img1, img2)
    sam_val = calculate_sam(img1, img2)
    ergas_val = calculate_ergas(img1, img2)

    # Process return values
    metrics = []
    for metric in [psnr, ssim, cc_val, sam_val, ergas_val]:
        metrics.append(metric.item() if torch.is_tensor(metric) else metric)

    return metrics


# -----------------------------------------------------------------------------
# Tester
# -----------------------------------------------------------------------------
class Tester:
    """Tester for the ProRecFusionNet edge-enhanced fusion model."""

    def __init__(self):
        """Initialize test environment."""
        # ──────────────── Test configuration ────────────────
        self.batch_size = 1
        self.model = ProRecFusionNet()

        # ──────────────── Dataset ────────────────
        self.test_dataset = load_testing_dataset(config.test_data_dir)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6
        )

        # ──────────────── Device ────────────────
        self.device = torch.device(
            "cuda" if config.cuda and torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.current_epoch = 0

        # ──────────────── Metrics tracking ────────────────
        self.metric_names = ["PSNR", "SSIM", "CC", "SAM", "ERGAS"]

        # ──────────────── Paths ────────────────
        self.record_dir = config.record_dir
        self.output_dir = config.output_dir

        # Ensure directories exist
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize test record file
        self.test_metrics_file = self.record_dir / "test_metrics.txt"

        # Log initialization
        logger.info(
            f"Initialized tester with device: {self.device}, dataset size: {len(self.test_dataset)}"
        )

    # -----------------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------------
    def load_checkpoint(self, model_path):
        """Load model checkpoint."""
        if not model_path.exists():
            error_msg = f"Pretrained model not found: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            logger.info(f"Loading model: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.current_epoch = checkpoint.get("epoch", 0)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            logger.info(f"Successfully loaded model, epoch: {self.current_epoch}")
        except Exception as e:
            error_msg = f"Failed to load model {model_path}: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    # -----------------------------------------------------------------------------
    # Image saving
    # -----------------------------------------------------------------------------
    def save_fusion_result(self, fusion_output, sample_id):
        """Save fusion result image."""
        # Convert and denormalize image
        image_np = fusion_output.detach().cpu().numpy()
        image_np = np.transpose(image_np, (0, 2, 3, 1))
        image_np = denormalize_image(image_np[0])
        image_to_save = image_np.transpose(2, 0, 1)

        # Debug image shape and range
        img_min, img_max = np.min(image_to_save), np.max(image_to_save)
        logger.debug(f"Image stats - shape: {image_to_save.shape}, range: [{img_min}, {img_max}]")

        # Save image using custom save_geotiff function
        output_path = self.output_dir / f"result_image_{sample_id}.tif"
        try:
            save_geotiff(
                str(output_path),
                [708573.6, 2549743.2],
                0.6,
                -0.6,
                image_to_save,
                4,
            )
            logger.debug(f"Saved image: {output_path}")  # Debug level to reduce output
        except Exception as e:
            logger.error(f"Failed to save image {output_path}: {e}")

    # -----------------------------------------------------------------------------
    # Logging helpers
    # -----------------------------------------------------------------------------
    def format_metrics_line(self, sample_id, metrics):
        """Format metrics as a readable line."""
        metrics_str = ", ".join([f"{name}: {value:.4f}" for name, value in zip(self.metric_names, metrics)])
        return f"Sample {sample_id}: {metrics_str}"

    def format_summary(self, metrics, processing_time, total_samples):
        """Format test summary."""
        separator = "=" * 50
        summary = [
            separator,
            "TEST SUMMARY",
            separator,
            f"Total samples: {total_samples}",
            f"Average processing time: {processing_time:.4f} seconds/sample",
            separator
        ]

        # Add metrics
        for name, value in zip(self.metric_names, metrics):
            summary.append(f"Average {name}: {value:.6f}")

        summary.append(separator)
        return "\n".join(summary)

    def log_metrics(self, sample_id, metrics, file, mode="a"):
        """Log metrics to file and logger."""
        metrics_line = self.format_metrics_line(sample_id, metrics)

        # Write to file
        with open(file, mode) as f:
            f.write(f"{metrics_line}\n")

        # Only log certain samples to avoid console spam
        total_samples = len(self.test_dataloader)
        log_interval = max(1, total_samples // 10)  # Log ~10 samples
        if sample_id % log_interval == 0 or sample_id == total_samples - 1:
            logger.info(metrics_line)

    # -----------------------------------------------------------------------------
    # Testing
    # -----------------------------------------------------------------------------
    def run_test(self):
        """Execute the complete testing procedure."""
        self.model.eval()
        self.load_checkpoint(config.pretrained_model_path)

        # Prepare metrics tracking
        metrics_data = {
            'time': [],
            'psnr': [],
            'ssim': [],
            'cc': [],
            'sam': [],
            'ergas': []
        }

        # Always create a new test log file to avoid confusion with old results
        # This addresses the issue where old 0.0 SAM values were showing up
        with open(self.test_metrics_file, "w") as f:
            f.write(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Using model: {config.pretrained_model_path}\n\n")

        # Log test start
        total_samples = len(self.test_dataloader)
        start_time = time.time()
        logger.info(f"Starting test on {total_samples} samples...")

        # Get first batch to inspect image format for debugging
        first_batch = next(iter(self.test_dataloader))
        pan, lr, lr_u, target = first_batch

        # Log data shapes and ranges to help diagnose metric issues
        pan_stats = (pan.shape, torch.min(pan).item(), torch.max(pan).item())
        lr_stats = (lr.shape, torch.min(lr).item(), torch.max(lr).item())
        target_stats = (target.shape, torch.min(target).item(), torch.max(target).item())

        logger.info(f"Data shapes and ranges:")
        logger.info(f"PAN: shape={pan_stats[0]}, range=[{pan_stats[1]:.2f}, {pan_stats[2]:.2f}]")
        logger.info(f"LR: shape={lr_stats[0]}, range=[{lr_stats[1]:.2f}, {lr_stats[2]:.2f}]")
        logger.info(f"Target: shape={target_stats[0]}, range=[{target_stats[1]:.2f}, {target_stats[2]:.2f}]")

        # Initialize progress bar
        progress_bar = tqdm(
            self.test_dataloader,
            desc="Testing progress",
            total=total_samples,
            unit="sample"
        )

        # Process each sample
        for sample_idx, (pan, lr, lr_u, target) in enumerate(progress_bar):
            # Time tracking and data preparation
            sample_start_time = time.time()
            pan = pan.to(self.device)
            lr = lr.to(self.device)
            lr_u = lr_u.to(self.device)
            target = target.to(self.device)

            # Forward pass
            with torch.no_grad():
                fusion_result = self.model(pan, lr)

            # Calculate metrics
            quality_metrics = compute_metrics(fusion_result, target)

            # Update progress bar with current metrics
            progress_bar.set_postfix({
                'PSNR': f"{quality_metrics[0]:.2f}",
                'SSIM': f"{quality_metrics[1]:.4f}",
                'SAM': f"{quality_metrics[3]:.4f}"  # Added SAM to progress display
            })

            # Store metrics
            for i, key in enumerate(['psnr', 'ssim', 'cc', 'sam', 'ergas']):
                metrics_data[key].append(quality_metrics[i])

            # Log metrics
            sample_id = sample_idx + 1
            self.log_metrics(sample_id, quality_metrics, self.test_metrics_file, "a")

            # Save output image
            self.save_fusion_result(fusion_result, sample_id)

            # Track processing time
            processing_time = time.time() - sample_start_time
            metrics_data['time'].append(processing_time)

        # Calculate average metrics
        average_metrics = [np.mean(metrics_data[key]) for key in ['psnr', 'ssim', 'cc', 'sam', 'ergas']]
        average_time = np.mean(metrics_data['time'])

        # Double check SAM average calculation
        sam_values = metrics_data['sam']
        sam_avg = np.mean(sam_values)
        logger.info(f"SAM values check - avg: {sam_avg:.6f}, min: {min(sam_values):.4f}, max: {max(sam_values):.4f}")

        # Generate and log summary
        summary = self.format_summary(average_metrics, average_time, total_samples)
        logger.info(summary)

        # Write summary to file
        with open(self.test_metrics_file, "a") as f:
            f.write(f"\n{summary}\n")

        # Log total test time
        total_test_time = time.time() - start_time
        logger.info(f"Test completed in {total_test_time:.2f} seconds")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main():
    """Main function."""
    tester = Tester()
    tester.run_test()


if __name__ == "__main__":
    main()