import pandas as pd
import numpy as np
import logging


INPUT_DATA_FILE = 'traffic_data.parquet' 
OUTPUT_AUGMENTED_FILE = 'augmented_traffic_data.parquet' 
NUMERICAL_COLS_TO_AUGMENT = ['speed', 'travel_time']


NUM_SYNTHETIC_SAMPLES_FACTOR = 0.5 
NOISE_LEVEL = 0.10 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
np.random.seed(42) 

# Data Loading
def load_real_data(filepath: str, cols_to_convert: list) -> pd.DataFrame:
    """Loads the real traffic data and ensures target columns are numeric."""
    logging.info(f"Loading real data from: {filepath}")
    try:
        df = pd.read_parquet(filepath)
        logging.info(f"Loaded real data shape: {df.shape}")

        # Added Conversion Step
        logging.info(f"Attempting to convert columns to numeric: {cols_to_convert}")
        for col in cols_to_convert:
            if col in df.columns:
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors='coerce')
                new_dtype = df[col].dtype
                nan_count = df[col].isna().sum()
                if original_dtype != new_dtype:
                     logging.info(f"Converted column '{col}' from {original_dtype} to {new_dtype}.")
                if nan_count > 0:
                    logging.warning(f"Column '{col}' contains {nan_count} non-numeric values (converted to NaN).")
            else:
                logging.warning(f"Column '{col}' specified for conversion not found in DataFrame.")


        return df
    except FileNotFoundError:
        logging.error(f"Real data file not found: {filepath}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading or converting real data: {e}")
        return pd.DataFrame()

# Augmentation Function: Noise Injection
def add_gaussian_noise(df: pd.DataFrame, target_columns: list[str], noise_level: float, num_samples: int) -> pd.DataFrame:

    if df.empty or num_samples <= 0:
        return pd.DataFrame()

    logging.info(f"Generating {num_samples} synthetic samples via noise injection...")

    synthetic_df = df.sample(n=num_samples, replace=True, random_state=42).copy()
    synthetic_df.reset_index(drop=True, inplace=True)

    for col in target_columns:
        if col not in df.columns:
            logging.warning(f"Column '{col}' not found in DataFrame. Skipping noise addition for this column.")
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
             logging.warning(f"Column '{col}' is not numeric type ({df[col].dtype}). Skipping noise addition.")
             continue

        std_dev = df[col].std()
        noise_magnitude = std_dev * noise_level

        if pd.isna(noise_magnitude) or noise_magnitude <= 0: 
             logging.warning(f"Could not add noise to column '{col}' (std dev is {std_dev}). Skipping.")
             continue

        noise = np.random.normal(loc=0.0, scale=noise_magnitude, size=num_samples)
        synthetic_df[col] = synthetic_df[col] + noise

        if col == 'speed' or col == 'travel_time':
            synthetic_df[col] = synthetic_df[col].clip(lower=0) 

        logging.info(f"Added noise to column '{col}' with std dev {noise_magnitude:.4f}")

    return synthetic_df

# Data Combination 
def combine_data(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Combines real and synthetic data into a single DataFrame with a flag."""
    if synthetic_df.empty:
        logging.warning("Synthetic dataframe is empty, returning only real data.")
        real_df['is_synthetic'] = False
        return real_df

    logging.info("Combining real and synthetic data...")
    synthetic_df['is_synthetic'] = True
    real_df['is_synthetic'] = False


    cols = real_df.columns.tolist()
    for col in cols:
         if col not in synthetic_df.columns and col != 'is_synthetic':
              pass 

    augmented_df = pd.concat([real_df, synthetic_df[cols]], ignore_index=True) 
    logging.info(f"Combined data shape: {augmented_df.shape}")
    return augmented_df

# Main Execution
if __name__ == "__main__":
    real_traffic_df = load_real_data(INPUT_DATA_FILE, NUMERICAL_COLS_TO_AUGMENT) 

    if real_traffic_df.empty:
        logging.error("Failed to load real data. Exiting.")
        exit()

    valid_numeric_cols = [
        col for col in NUMERICAL_COLS_TO_AUGMENT
        if col in real_traffic_df.columns and pd.api.types.is_numeric_dtype(real_traffic_df[col])
    ]
    if len(valid_numeric_cols) < len(NUMERICAL_COLS_TO_AUGMENT):
        logging.warning(f"After conversion, only these columns are numeric and will be augmented: {valid_numeric_cols}")


    num_synthetic = int(len(real_traffic_df) * NUM_SYNTHETIC_SAMPLES_FACTOR)
    if num_synthetic <= 0:
        logging.warning("Number of synthetic samples to generate is zero. No augmentation performed.")
        augmented_dataset = real_traffic_df
        augmented_dataset['is_synthetic'] = False
    else:
        synthetic_traffic_df = add_gaussian_noise(
            real_traffic_df,
            valid_numeric_cols, 
            NOISE_LEVEL,
            num_synthetic
        )

        augmented_dataset = combine_data(real_traffic_df, synthetic_traffic_df)

    if not augmented_dataset.empty:
        try:
            augmented_dataset.to_parquet(OUTPUT_AUGMENTED_FILE, engine='pyarrow', index=False)
            logging.info(f"Augmented dataset saved to: {OUTPUT_AUGMENTED_FILE}")
        except Exception as e:
            logging.error(f"Failed to save augmented data: {e}")