import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import diffusers # Required import
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import logging
import os
import re 
from PIL import Image


INPUT_DATA_FILE = 'augmented_traffic_data.parquet' 
GRID_RESOLUTION = 64
IN_CHANNELS = 1 
OUT_CHANNELS = 1
BOUNDING_BOX = (-74.02, 40.70, -73.93, 40.83)
MIN_SPEED = 0.0
MAX_SPEED = 65.0 

# Model Configuration
# See: https://huggingface.co/docs/diffusers/api/models/unet2d#diffusers.UNet2DModel
MODEL_CONFIG = {
    "sample_size": GRID_RESOLUTION,
    "in_channels": IN_CHANNELS,
    "out_channels": OUT_CHANNELS,
    "layers_per_block": 2,
    "block_out_channels": (64, 128, 128, 256),
    "down_block_types": (
        "DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
    ),
    "up_block_types": (
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D",
    ),
}

# Training Configuration
NUM_TRAINING_STEPS = 10000
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LR_WARMUP_STEPS = 500
GRADIENT_ACCUMULATION_STEPS = 1
SAVE_IMAGE_STEPS = 1000
SAVE_MODEL_STEPS = 5000
OUTPUT_DIR = "diffusion_output"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_midpoint_coord(link_points_str):
    """
    Parses 'link_points' string for midpoint coordinates.
    Returns (None, None) on failure.
    """
    if pd.isna(link_points_str) or not isinstance(link_points_str, str):
        return None, None
    points = re.findall(r'(-?\d+\.\d+)[,\s](-?\d+\.\d+)', link_points_str)
    if not points:
        return None, None
    try:
        lat1, lon1 = map(float, points[0])
        lat2, lon2 = map(float, points[-1])
        if not (-90 <= lat1 <= 90 and -180 <= lon1 <= 180 and -90 <= lat2 <= 90 and -180 <= lon2 <= 180):
             return None, None
        return (lat1 + lat2) / 2, (lon1 + lon2) / 2
    except (ValueError, IndexError):
        return None, None

# Data Preprocessing Function
def load_and_preprocess_grid_data(data_file: str, resolution: int, channels: int) -> np.ndarray:
    if channels != 1:
        logging.error(f"This implementation currently only supports channels=1. Got {channels}.")
        return np.array([])

    logging.info(f"Loading and preprocessing data from {data_file} for grid {resolution}x{resolution}...")
    try:
        df = pd.read_parquet(data_file)
        logging.info(f"Loaded raw data shape: {df.shape}")

        if 'data_as_of' not in df.columns or 'speed' not in df.columns or 'link_points' not in df.columns:
             logging.error("Input dataframe missing required columns: 'data_as_of', 'speed', 'link_points'.")
             return np.array([])

        df['data_as_of'] = pd.to_datetime(df['data_as_of'], errors='coerce')
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        df.dropna(subset=['speed', 'data_as_of', 'link_points'], inplace=True)
        logging.info(f"Data shape after initial cleaning: {df.shape}")
        if df.empty:
            logging.error("No valid data after initial cleaning.")
            return np.array([])

        logging.info("Extracting midpoint coordinates from 'link_points'...")
        coords = df['link_points'].astype(str).apply(get_midpoint_coord)
        df['latitude'] = coords.apply(lambda x: x[0] if x else None)
        df['longitude'] = coords.apply(lambda x: x[1] if x else None)
        df.dropna(subset=['latitude', 'longitude'], inplace=True)
        logging.info(f"Data shape after coordinate extraction: {df.shape}")
        if df.empty:
            logging.error("No valid coordinates found or extracted.")
            return np.array([])

        min_lon, min_lat, max_lon, max_lat = BOUNDING_BOX
        df = df[
            (df['longitude'] >= min_lon) & (df['longitude'] <= max_lon) &
            (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat)
        ]
        logging.info(f"Data shape after filtering by bounding box {BOUNDING_BOX}: {df.shape}")
        if df.empty:
            logging.error("No data found within the specified bounding box.")
            return np.array([])

        lat_rng = max_lat - min_lat
        lon_rng = max_lon - min_lon
        if lat_rng <= 0 or lon_rng <= 0:
            logging.error("Invalid bounding box dimensions.")
            return np.array([])

        df['grid_i'] = (((df['latitude'] - min_lat) / lat_rng) * resolution).astype(int).clip(0, resolution - 1)
        df['grid_j'] = (((df['longitude'] - min_lon) / lon_rng) * resolution).astype(int).clip(0, resolution - 1)
        df['time_hour'] = df['data_as_of'].dt.floor('h') 
        logging.info("Aggregating mean speed per grid cell per hour...")
        grid_agg = df.groupby(['time_hour', 'grid_i', 'grid_j'], observed=False)['speed'].mean()

        all_hours = grid_agg.index.get_level_values('time_hour').unique()
        all_i = np.arange(resolution)
        all_j = np.arange(resolution)
        multi_index = pd.MultiIndex.from_product([all_hours, all_i, all_j], names=['time_hour', 'grid_i', 'grid_j'])
        grid_complete = grid_agg.reindex(multi_index, fill_value=MIN_SPEED)
        num_samples = len(all_hours)
        grid_complete = grid_complete.sort_index()
        all_grids = grid_complete.values.reshape(num_samples, resolution, resolution)
        all_grids = np.expand_dims(all_grids, axis=1)
        logging.info(f"Created spatio-temporal grids. Shape: {all_grids.shape}")

        if MAX_SPEED <= MIN_SPEED:
             logging.error("MAX_SPEED must be greater than MIN_SPEED for normalization.")
             return np.array([])
        normalized_grids = ((all_grids.astype(np.float32) - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)) * 2.0 - 1.0
        normalized_grids = np.clip(normalized_grids, -1.0, 1.0)
        logging.info("Normalized grid data to [-1, 1].")

        return normalized_grids.astype(np.float32)

    except FileNotFoundError:
        logging.error(f"Data file not found: {data_file}")
        return np.array([])
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}", exc_info=True)
        return np.array([])

# PyTorch Dataset
class UrbanGridDataset(Dataset):
    def __init__(self, grid_data: np.ndarray):
        self.grid_data = torch.from_numpy(grid_data.astype(np.float32))

    def __len__(self):
        return len(self.grid_data)

    def __getitem__(self, idx):
        return self.grid_data[idx]

def train_diffusion(
    model_config,
    grid_data,
    num_training_steps,
    batch_size,
    learning_rate,
    lr_warmup_steps,
    gradient_accumulation_steps,
    output_dir,
    save_image_steps,
    save_model_steps
):
    accelerator = Accelerator(
        mixed_precision="no", 
        gradient_accumulation_steps=gradient_accumulation_steps,
        project_dir=os.path.join(output_dir, "logs")
    )
    logging.info(f"Using device: {accelerator.device}")

    dataset = UrbanGridDataset(grid_data)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNet2DModel(**model_config)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    global_step = 0
    progress_bar = tqdm(total=num_training_steps, disable=not accelerator.is_main_process, desc="Training Steps")

    while global_step < num_training_steps:
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for step, batch in enumerate(train_dataloader):
            clean_images = batch
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise) 

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                 progress_bar.update(1)
                 global_step += 1
                 logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
                 progress_bar.set_postfix(logs)

            if accelerator.is_main_process:
                if global_step > 0 and global_step % save_image_steps == 0:
                    logging.info(f"Generating sample images at step {global_step}...")
                    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                    with torch.no_grad(): 
                        eval_images = pipeline(batch_size=4, generator=torch.manual_seed(global_step)).images

                    save_dir = os.path.join(output_dir, "samples")
                    for i, img in enumerate(eval_images):
                        try:
                            img.save(f"{save_dir}/sample_{global_step}_{i}.png")
                        except Exception as e:
                            logging.error(f"Failed to save sample image {i} at step {global_step}: {e}")
                    logging.info(f"Saved {len(eval_images)} sample images to {save_dir}")

                if global_step > 0 and global_step % save_model_steps == 0:
                    save_path = os.path.join(output_dir, f"checkpoint_{global_step}")
                    accelerator.save_state(save_path)
                    logging.info(f"Saved model checkpoint to {save_path} at step {global_step}")

            if global_step >= num_training_steps:
                break

    logging.info("Training finished.")
    if accelerator.is_main_process:
        final_save_path = os.path.join(output_dir, "final_checkpoint")
        accelerator.save_state(final_save_path)
        logging.info(f"Saved final model state to {final_save_path}")


# Main Execution
if __name__ == "__main__":
    grid_data_np = load_and_preprocess_grid_data(INPUT_DATA_FILE, GRID_RESOLUTION, IN_CHANNELS)

    if grid_data_np is None or grid_data_np.size == 0:
        logging.error("Preprocessing returned no data. Exiting.")
        exit()

    if len(grid_data_np.shape) != 4 or \
       grid_data_np.shape[1] != IN_CHANNELS or \
       grid_data_np.shape[2] != GRID_RESOLUTION or \
       grid_data_np.shape[3] != GRID_RESOLUTION:
        logging.error(f"Preprocessed data has incorrect shape {grid_data_np.shape}. "
                      f"Expected (num_samples, {IN_CHANNELS}, {GRID_RESOLUTION}, {GRID_RESOLUTION}).")
        exit()
    logging.info(f"Preprocessing successful. Data shape for training: {grid_data_np.shape}")

    train_diffusion(
        model_config=MODEL_CONFIG,
        grid_data=grid_data_np,
        num_training_steps=NUM_TRAINING_STEPS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_warmup_steps=LR_WARMUP_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        output_dir=OUTPUT_DIR,
        save_image_steps=SAVE_IMAGE_STEPS,
        save_model_steps=SAVE_MODEL_STEPS
    )