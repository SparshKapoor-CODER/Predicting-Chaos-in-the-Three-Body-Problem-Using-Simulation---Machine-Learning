# sequence_modeling.py 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path

# Configuration
CONFIG = {
    "data_path": "three_body_dataset.csv",
    "model_save_path": "models/three_body_lstm.pth",
    "seq_length": 50,          # Number of time steps in each sequence
    "pred_length": 10,         # Number of steps to predict ahead
    "batch_size": 64,
    "hidden_size": 128,
    "num_layers": 2,
    "learning_rate": 0.001,
    "num_epochs": 20,
    "train_ratio": 0.8,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# Create directories
Path(os.path.dirname(CONFIG["model_save_path"])).mkdir(parents=True, exist_ok=True)

class ThreeBodyDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

class TrajectoryPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TrajectoryPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(CONFIG["device"])
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(CONFIG["device"])
        else:
            h0, c0 = hidden
        
        # LSTM forward pass
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Only use the last time step's output for prediction
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Pass through FC layers
        predictions = self.fc(last_output)  # Shape: (batch_size, output_size)
        
        # Reshape predictions to match target shape (batch_size, pred_length, output_features)
        predictions = predictions.view(batch_size, CONFIG["pred_length"], -1)
        
        return predictions

def prepare_sequence_data(df):
    """Convert raw data into sequence format for LSTM"""
    # Extract features (positions and velocities for all bodies)
    features = df[[
        'p1_x', 'p1_y', 'p1_z', 'p2_x', 'p2_y', 'p2_z', 'p3_x', 'p3_y', 'p3_z',
        'v1_x', 'v1_y', 'v1_z', 'v2_x', 'v2_y', 'v2_z', 'v3_x', 'v3_y', 'v3_z'
    ]].values
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Create sequences - assuming each row is consecutive in time
    sequences = []
    targets = []
    
    for i in range(len(features) - CONFIG["seq_length"] - CONFIG["pred_length"] + 1):
        seq = features[i:i + CONFIG["seq_length"]]
        target = features[i + CONFIG["seq_length"]:i + CONFIG["seq_length"] + CONFIG["pred_length"]]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def train_model(train_loader, val_loader, input_size, output_size):
    model = TrajectoryPredictor(
        input_size=input_size,
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        output_size=output_size * CONFIG["pred_length"]  # We predict pred_length steps at once
    ).to(CONFIG["device"])
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(CONFIG["num_epochs"]):
        model.train()
        epoch_train_loss = 0
        
        for sequences, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            sequences = sequences.to(CONFIG["device"])
            targets = targets.to(CONFIG["device"])
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(CONFIG["device"])
                targets = targets.to(CONFIG["device"])
                
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CONFIG["model_save_path"])
            print("Saved new best model")
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.show()
    plt.close()
    
    return model

def evaluate_model(model, test_loader, output_size):
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    
    # Collect samples for visualization
    sample_inputs = []
    sample_targets = []
    sample_outputs = []
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences = sequences.to(CONFIG["device"])
            targets = targets.to(CONFIG["device"])
            
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # Save some samples for visualization
            if len(sample_inputs) < 3:
                sample_inputs.append(sequences[0].cpu().numpy())
                sample_targets.append(targets[0].cpu().numpy())
                sample_outputs.append(outputs[0].cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    print(f"\nTest Loss: {avg_loss:.4f}")
    
    # Visualize predictions
    visualize_predictions(sample_inputs, sample_targets, sample_outputs, output_size)
    
    return avg_loss

def visualize_predictions(inputs, targets, outputs, output_size):
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    body_names = ['Body 1', 'Body 2', 'Body 3']
    coord_names = ['X', 'Y', 'Z']
    
    for i in range(3):  # For each sample
        for j in range(3):  # For each coordinate
            ax = axes[i, j]
            
            # Plot history
            history = inputs[i][:, j::3]  # Get coordinate j for all bodies
            for k in range(3):  # For each body
                ax.plot(range(CONFIG["seq_length"]), history[:, k], 
                       label=f'{body_names[k]} History')
            
            # Plot true future
            future = targets[i][:, j::3]
            for k in range(3):
                ax.plot(range(CONFIG["seq_length"], CONFIG["seq_length"] + CONFIG["pred_length"]), 
                       future[:, k], '--', label=f'{body_names[k]} True')
            
            # Plot predicted future
            pred = outputs[i][:, j::3]
            for k in range(3):
                ax.plot(range(CONFIG["seq_length"], CONFIG["seq_length"] + CONFIG["pred_length"]), 
                       pred[:, k], ':', label=f'{body_names[k]} Predicted')
            
            ax.set_title(f'Sample {i+1} - {coord_names[j]} Coordinate')
            ax.set_xlabel('Time Step')
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    plt.close()

def main():
    # Load and prepare data
    df = pd.read_csv(CONFIG["data_path"])
    sequences, targets = prepare_sequence_data(df)
    
    # Split into train/val/test
    num_samples = len(sequences)
    train_size = int(CONFIG["train_ratio"] * num_samples)
    val_size = int(0.1 * num_samples)
    
    train_seq = sequences[:train_size]
    train_tgt = targets[:train_size]
    
    val_seq = sequences[train_size:train_size+val_size]
    val_tgt = targets[train_size:train_size+val_size]
    
    test_seq = sequences[train_size+val_size:]
    test_tgt = targets[train_size+val_size:]
    
    # Create datasets and loaders
    train_dataset = ThreeBodyDataset(train_seq, train_tgt)
    val_dataset = ThreeBodyDataset(val_seq, val_tgt)
    test_dataset = ThreeBodyDataset(test_seq, test_tgt)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"])
    
    # Get input/output sizes
    input_size = train_seq.shape[2]  # Number of features at each time step
    output_size = train_tgt.shape[2]  # Features per time step
    
    # Train model
    print("Starting training...")
    model = train_model(train_loader, val_loader, input_size, output_size)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss = evaluate_model(model, test_loader, output_size)
    
    print("\nTraining complete!")
    print(f"Best model saved to: {CONFIG['model_save_path']}")
    print(f"Final test loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()