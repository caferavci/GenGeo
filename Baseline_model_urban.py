# baseline_lstm.py (Corrected)
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# import joblib # Uncomment if you want to save scalers

# --- Configuration ---
DATA_FILE = 'traffic_data.parquet' # Assumes the parquet file from your pipeline is in the same directory
TARGET_COLUMN = 'speed' # The column you want to predict
# Feature columns (adjust based on your actual Parquet file columns after processing)
FEATURE_COLUMNS = ['borough_code', 'encoded_direction', 'link_id_encoded', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
SEQUENCE_LENGTH = 5 # Number of past time steps to use for prediction
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
TEST_SIZE = 0.2 # Fraction of data for validation/testing

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Preprocessing and Loading (Corrected) ---
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing: Feature engineering, scaling.
    You MUST adapt this function based on your specific data and feature needs.
    """
    logging.info("Starting preprocessing...")

    # Convert date column
    df['data_as_of'] = pd.to_datetime(df['data_as_of'])

    # Feature Engineering (Example: time features)
    df['hour'] = df['data_as_of'].dt.hour
    df['dayofweek'] = df['data_as_of'].dt.dayofweek

    # --- CORRECTED CYCLICAL ENCODING ---
    # Convert pandas Series to PyTorch Tensor before applying torch functions
    hour_tensor = torch.tensor(df['hour'].values, dtype=torch.float32)
    dayofweek_tensor = torch.tensor(df['dayofweek'].values, dtype=torch.float32)

    # Apply torch functions and convert result back to numpy array to store in DataFrame
    df['hour_sin'] = torch.sin(2 * torch.pi * hour_tensor / 24.0).numpy()
    df['hour_cos'] = torch.cos(2 * torch.pi * hour_tensor / 24.0).numpy()
    df['dayofweek_sin'] = torch.sin(2 * torch.pi * dayofweek_tensor / 7.0).numpy()
    df['dayofweek_cos'] = torch.cos(2 * torch.pi * dayofweek_tensor / 7.0).numpy()
    # --- END CORRECTION ---


    # Example: Encode categorical features (replace with actual relevant columns)
    # Ensure these columns exist or adapt as needed
    if 'borough' in df.columns:
        df['borough_code'] = df['borough'].astype('category').cat.codes
    else:
         logging.warning("Column 'borough' not found, using placeholder 0.")
         df['borough_code'] = 0 # Placeholder if column missing

    if 'encoded_direction' in df.columns:
        # Assuming 'encoded_direction' might be categorical like 'NB', 'SB' etc.
        df['encoded_direction'] = df['encoded_direction'].astype('category').cat.codes
    else:
         logging.warning("Column 'encoded_direction' not found, using placeholder 0.")
         df['encoded_direction'] = 0 # Placeholder

    if 'link_id' in df.columns:
        df['link_id_encoded'] = df['link_id'].astype('category').cat.codes
    else:
         logging.warning("Column 'link_id' not found, using placeholder 0.")
         df['link_id_encoded'] = 0 # Placeholder

    # --- Crucial Step: Scaling ---
    # Scale features and target separately for potential inverse transform later
    feature_scaler = StandardScaler()
    # Ensure all FEATURE_COLUMNS exist before trying to scale them
    valid_feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
    if not valid_feature_cols:
        raise ValueError("None of the specified FEATURE_COLUMNS were found in the DataFrame after preprocessing.")

    # Scale only existing feature columns
    df[valid_feature_cols] = feature_scaler.fit_transform(df[valid_feature_cols])

    # Ensure TARGET_COLUMN exists before scaling
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    target_scaler = StandardScaler()
    df[[TARGET_COLUMN]] = target_scaler.fit_transform(df[[TARGET_COLUMN]])
    # Note: Save these scalers if you need to inverse_transform predictions later!
    # Example: joblib.dump(feature_scaler, 'feature_scaler.joblib')
    # Example: joblib.dump(target_scaler, 'target_scaler.joblib')

    logging.info(f"Preprocessing complete. Using features: {valid_feature_cols}")
    return df, valid_feature_cols # Return valid columns used


class TrafficSequenceDataset(Dataset):
    """
    Dataset class to create sequences for LSTM.
    Assumes data is sorted by time for each entity (e.g., link_id).
    """
    def __init__(self, data: pd.DataFrame, feature_cols: list, target_col: str, sequence_length: int):
        self.sequence_length = sequence_length
        self.feature_cols = feature_cols
        self.target_col = target_col

        # Ensure data is sorted appropriately (e.g., by link_id and then time)
        # This step is CRITICAL for creating meaningful sequences.
        # Example sorting: You might need to group by 'link_id' or similar identifier.
        # data = data.sort_values(by=['link_id', 'data_as_of'])

        # Convert relevant DataFrame columns to tensors directly
        self.features = torch.tensor(data[self.feature_cols].values, dtype=torch.float32)
        self.target = torch.tensor(data[self.target_col].values, dtype=torch.float32)

    def __len__(self):
        # Subtract sequence_length because each item needs 'sequence_length' historical points
        # Also subtract 1 because the target is at index `idx + sequence_length`
        return len(self.features) - self.sequence_length

    def __getitem__(self, idx):
        # Sequence of features from idx to idx + sequence_length - 1
        feature_sequence = self.features[idx : idx + self.sequence_length]
        # Target is the value immediately after the sequence end
        target_value = self.target[idx + self.sequence_length]
        return feature_sequence, target_value

# --- Model Definition ---
class BaselineLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 1, output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM layer: batch_first=True means input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Linear layer to map the hidden state output to the desired output dimension
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Initialize hidden state and cell state with zeros
        # Shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Pass through LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # We only need the output from the last time step for prediction
        last_time_step_out = out[:, -1, :]

        # Pass the last time step's output through the fully connected layer
        prediction = self.fc(last_time_step_out)
        # If output_dim is 1, squeeze the last dimension for loss calculation
        if prediction.shape[-1] == 1:
            return prediction.squeeze(-1)
        else:
            return prediction


# --- Training Loop ---
def train_model(model, train_loader, val_loader, epochs, learning_rate, device):
    """Basic training loop."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss() # Mean Squared Error for regression

    logging.info("Starting training...")
    for epoch in range(epochs):
        model.train() # Set model to training mode
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Forward pass
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0) # Accumulate loss weighted by batch size

        train_loss /= len(train_loader.dataset) # Average loss over the dataset

        # Validation phase
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = model(batch_x)
                loss = loss_fn(pred, batch_y)
                val_loss += loss.item() * batch_x.size(0) # Accumulate loss weighted by batch size

        val_loss /= len(val_loader.dataset) # Average loss over the dataset

        logging.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    logging.info("Training complete.")
    # Consider saving the model
    # torch.save(model.state_dict(), 'baseline_lstm_model.pth')

# --- Main Execution ---
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load data
    try:
        raw_df = pd.read_parquet(DATA_FILE)
        logging.info(f"Loaded data: {raw_df.shape}")
        # Optional: Downsample for faster testing/debugging
        # raw_df = raw_df.sample(frac=0.1, random_state=42)
        # logging.info(f"Downsampled data to: {raw_df.shape}")
    except FileNotFoundError:
        logging.error(f"Error: Data file '{DATA_FILE}' not found. Make sure it's in the correct directory.")
        exit()
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        exit()

    # Preprocess data (Adapt this function heavily!)
    try:
        # IMPORTANT: Ensure data is sorted by time, possibly grouped by location ID, BEFORE splitting/sequencing
        # Example: raw_df = raw_df.sort_values(by=['link_id', 'data_as_of'])
        processed_df, actual_feature_cols = preprocess_data(raw_df.copy())
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}", exc_info=True)
        exit()

    # Split data (ensure shuffling is appropriate or handle time series splits carefully)
    # Using chronological split is generally better for time series.
    # Calculate split index
    split_idx = int(len(processed_df) * (1 - TEST_SIZE))
    train_df = processed_df.iloc[:split_idx]
    val_df = processed_df.iloc[split_idx:]
    # If you sorted by groups (e.g., link_id), ensure groups aren't split across train/val

    logging.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}")


    # Create datasets and dataloaders
    train_dataset = TrafficSequenceDataset(train_df, actual_feature_cols, TARGET_COLUMN, SEQUENCE_LENGTH)
    val_dataset = TrafficSequenceDataset(val_df, actual_feature_cols, TARGET_COLUMN, SEQUENCE_LENGTH)

    # Check if datasets are empty
    if len(train_dataset) == 0 or len(val_dataset) == 0:
         logging.error("Created datasets are empty. Check sequence length and data size after split/preprocessing.")
         exit()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # Shuffle training data
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False) # No shuffle for validation

    # Initialize model
    input_dim = len(actual_feature_cols) # Number of features used
    model = BaselineLSTM(input_dim=input_dim)

    # Train model
    train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device)