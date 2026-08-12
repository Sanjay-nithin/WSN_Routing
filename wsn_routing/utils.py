import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

# Define class mappings
LABEL_MAPPING = {
    'Healthy': 0,
    'Congested': 1,
    'Unhealthy': 2
}

INV_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

class WSNDataset(Dataset):
    """Custom PyTorch Dataset for WSN Node data."""
    def __init__(self, X, y):
        # Convert X to float32 tensor
        self.X = torch.tensor(X, dtype=torch.float32)
        # Convert y to long (integer) tensor for CrossEntropyLoss
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data(file_path='dataset.xlsx'):
    """
    Loads dataset.xlsx, extracts features and target, and maps target classes to integers.
    Returns:
        X (np.ndarray): raw feature array
        y (np.ndarray): target integer labels
        feature_names (list): names of the features used
    """
    # Load Excel file
    df = pd.read_excel(file_path)
    
    # Define target column
    target_col = 'Node_Status'
    
    # Identify feature columns (exclude Node_ID and target_col)
    feature_cols = [col for col in df.columns if col not in ['Node_ID', target_col]]
    
    # Extract features and targets
    X = df[feature_cols].values.astype(np.float32)
    
    # Map target strings to integers
    y = df[target_col].map(LABEL_MAPPING).values.astype(np.int64)
    
    return X, y, feature_cols

def prepare_dataloaders(X, y, test_size=0.2, batch_size=32, random_state=42):
    """
    Splits the dataset into train and test sets, normalizes features using StandardScaler,
    and returns PyTorch DataLoaders and scaler.
    """
    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Fit scaler on training data and transform both train and test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create Dataset instances
    train_dataset = WSNDataset(X_train_scaled, y_train)
    test_dataset = WSNDataset(X_test_scaled, y_test)
    
    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, test_loader, scaler, (X_train_scaled, X_test_scaled, y_train, y_test)
