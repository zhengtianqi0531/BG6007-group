import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


# ============================================
# MODEL DEFINITION
# ============================================
class HybridOximetryModel(nn.Module):
    def __init__(self, input_channels, window_size=300):
        super(HybridOximetryModel, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        conv_output_size = 64 * 75

        self.fc1 = nn.Linear(conv_output_size, 128)
        self.fc2 = nn.Linear(128, 64)

        self.regression_head = nn.Linear(64, 1)
        self.classification_head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        reg_output = self.regression_head(x)
        cls_output = self.classification_head(x)

        return reg_output, cls_output


# ============================================
# DATASET CLASS
# ============================================
class OximetryDataset(Dataset):
    def __init__(self, ppg_file, gt_file, window_size=300):
        self.ppg_df = pd.read_csv(ppg_file)
        self.gt_df = pd.read_csv(gt_file)

        self.data = pd.merge(
            self.ppg_df,
            self.gt_df,
            left_index=True,
            right_index=True,
            how='inner'
        )

        self.window_size = window_size
        ppg_columns = ['R', 'G', 'B']

        # Normalize
        self.data[ppg_columns] = (
                                         self.data[ppg_columns] - self.data[ppg_columns].mean()
                                 ) / self.data[ppg_columns].std()

        self.ppg_columns = ppg_columns

        # Find SpO2 column
        if 'SpO2_gt' in self.data.columns:
            self.spo2_column = 'SpO2_gt'
        elif 'sp02_1' in self.data.columns:
            self.spo2_column = 'sp02_1'
        elif 'SpO2 1' in self.data.columns:
            self.spo2_column = 'SpO2 1'
        else:
            spo2_cols = [col for col in self.data.columns
                         if 'spo2' in col.lower() or 'sp02' in col.lower()]
            if spo2_cols:
                self.spo2_column = spo2_cols[0]
            else:
                raise ValueError("SpO2 column not found!")

        print(f"Dataset loaded: {len(self.data)} rows, {len(self)} samples")
        print(f"Using SpO2 column: {self.spo2_column}")

    def __len__(self):
        return len(self.data) - self.window_size + 1

    def __getitem__(self, idx):
        ppg_window = self.data[self.ppg_columns].iloc[
                     idx: idx + self.window_size].values

        spo2_gt = self.data[self.spo2_column].iloc[idx + self.window_size - 1]

        classification_target = 1.0 if spo2_gt < 90 else 0.0
        ppg_tensor = torch.tensor(ppg_window, dtype=torch.float32)
        ppg_tensor = ppg_tensor.permute(1, 0)

        regression_target = torch.tensor(spo2_gt, dtype=torch.float32)
        classification_target = torch.tensor(classification_target, dtype=torch.float32)

        return ppg_tensor, regression_target, classification_target


# ============================================
# TRAINER CLASS (FIXED)
# ============================================
class ModelTrainer:
    def __init__(self, model, device, learning_rate=0.001):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Loss functions
        self.regression_loss_fn = nn.MSELoss()
        self.classification_loss_fn = nn.BCELoss()

        # History
        self.history = {
            'train_loss': [],
            'train_reg_loss': [],
            'train_cls_loss': [],
            'val_loss': [],
            'val_reg_loss': [],
            'val_cls_loss': []
        }

    def train_epoch(self, train_loader, reg_weight=1.0, cls_weight=1.0):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0

        pbar = tqdm(train_loader, desc="Training")
        for ppg, reg_target, cls_target in pbar:
            ppg = ppg.to(self.device)
            reg_target = reg_target.to(self.device).unsqueeze(1)
            cls_target = cls_target.to(self.device).unsqueeze(1)

            # Forward pass
            self.optimizer.zero_grad()
            reg_output, cls_output = self.model(ppg)

            # IMPORTANT: Clamp classification output to prevent NaN
            cls_output = torch.clamp(cls_output, min=1e-7, max=1 - 1e-7)

            # Calculate losses
            reg_loss = self.regression_loss_fn(reg_output, reg_target)
            cls_loss = self.classification_loss_fn(cls_output, cls_target)

            # Check for NaN and skip if found
            if torch.isnan(reg_loss) or torch.isnan(cls_loss):
                print(f"\nWarning: NaN detected, skipping batch")
                continue

            # Combined loss
            loss = reg_weight * reg_loss + cls_weight * cls_loss

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            total_reg_loss += reg_loss.item()
            total_cls_loss += cls_loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'reg': f'{reg_loss.item():.4f}',
                'cls': f'{cls_loss.item():.4f}'
            })

        n_batches = len(train_loader)
        return total_loss / n_batches, total_reg_loss / n_batches, total_cls_loss / n_batches

    def validate(self, val_loader, reg_weight=1.0, cls_weight=1.0):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_reg_loss = 0
        total_cls_loss = 0

        with torch.no_grad():
            for ppg, reg_target, cls_target in val_loader:
                ppg = ppg.to(self.device)
                reg_target = reg_target.to(self.device).unsqueeze(1)
                cls_target = cls_target.to(self.device).unsqueeze(1)

                reg_output, cls_output = self.model(ppg)

                # Clamp classification output
                cls_output = torch.clamp(cls_output, min=1e-7, max=1 - 1e-7)

                reg_loss = self.regression_loss_fn(reg_output, reg_target)
                cls_loss = self.classification_loss_fn(cls_output, cls_target)

                # Skip NaN
                if torch.isnan(reg_loss) or torch.isnan(cls_loss):
                    continue

                loss = reg_weight * reg_loss + cls_weight * cls_loss

                total_loss += loss.item()
                total_reg_loss += reg_loss.item()
                total_cls_loss += cls_loss.item()

        n_batches = len(val_loader)
        return total_loss / n_batches, total_reg_loss / n_batches, total_cls_loss / n_batches

    def train(self, train_loader, val_loader, epochs, reg_weight=1.0, cls_weight=1.0):
        """Full training loop"""
        print(f"\n{'=' * 70}")
        print(f"Starting Training")
        print(f"{'=' * 70}")
        print(f"Epochs: {epochs}")
        print(f"Regression weight: {reg_weight}")
        print(f"Classification weight: {cls_weight}")
        print(f"Device: {self.device}")

        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 70)

            # Train
            train_loss, train_reg, train_cls = self.train_epoch(
                train_loader, reg_weight, cls_weight
            )

            # Validate
            val_loss, val_reg, val_cls = self.validate(
                val_loader, reg_weight, cls_weight
            )

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_reg_loss'].append(train_reg)
            self.history['train_cls_loss'].append(train_cls)
            self.history['val_loss'].append(val_loss)
            self.history['val_reg_loss'].append(val_reg)
            self.history['val_cls_loss'].append(val_cls)

            # Print summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} (Reg: {train_reg:.4f}, Cls: {train_cls:.4f})")
            print(f"  Val Loss:   {val_loss:.4f} (Reg: {val_reg:.4f}, Cls: {val_cls:.4f})")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('oximetry_model_best.pth', epoch, val_loss)
                print(f"  ✓ Best model saved!")

        # Save final model
        self.save_model('oximetry_model_trained.pth', epochs - 1, val_loss)
        print(f"\n✓ Training complete! Final model saved.")

    def save_model(self, filepath, epoch, loss):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'history': self.history
        }, filepath)

    def plot_history(self, save_path='training_history.png'):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Total loss
        axes[0].plot(self.history['train_loss'], label='Train', linewidth=2)
        axes[0].plot(self.history['val_loss'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Loss')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Regression loss
        axes[1].plot(self.history['train_reg_loss'], label='Train', linewidth=2)
        axes[1].plot(self.history['val_reg_loss'], label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Regression Loss (MSE)')
        axes[1].set_title('SpO2 Regression Loss')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # Classification loss
        axes[2].plot(self.history['train_cls_loss'], label='Train', linewidth=2)
        axes[2].plot(self.history['val_cls_loss'], label='Validation', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Classification Loss (BCE)')
        axes[2].set_title('Hypoxemia Classification Loss')
        axes[2].legend()
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Training history plot saved to: {save_path}")
        plt.close()


# ============================================
# MAIN TRAINING SCRIPT
# ============================================
def main():
    print("=" * 70)
    print("OXIMETRY MODEL TRAINING (FIXED)")
    print("=" * 70)

    # Configuration
    PPG_FILE = '100002.csv'
    GT_FILE = '100002 (1).csv'
    WINDOW_SIZE = 300
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001  # Lower learning rate for stability
    VALIDATION_SPLIT = 0.2

    # Loss weights
    REG_WEIGHT = 1.0
    CLS_WEIGHT = 5.0  # Reduced from 10.0 for stability

    # Device - USE CPU TO AVOID CUDA ERROR
    device = torch.device('cpu')  # Changed from cuda to cpu
    print(f"\nUsing device: {device}")
    print("(Using CPU to avoid CUDA errors - slower but more stable)")

    # Check files exist
    if not os.path.exists(PPG_FILE):
        print(f"✗ Error: {PPG_FILE} not found!")
        return
    if not os.path.exists(GT_FILE):
        print(f"✗ Error: {GT_FILE} not found!")
        return

    # Load dataset
    print(f"\n{'=' * 70}")
    print("Loading Dataset")
    print(f"{'=' * 70}")

    dataset = OximetryDataset(PPG_FILE, GT_FILE, WINDOW_SIZE)

    # Split into train/val
    val_size = int(len(dataset) * VALIDATION_SPLIT)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    print(f"\nDataset split:")
    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Create model
    print(f"\n{'=' * 70}")
    print("Creating Model")
    print(f"{'=' * 70}")

    model = HybridOximetryModel(input_channels=3, window_size=WINDOW_SIZE)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel architecture:")
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = ModelTrainer(model, device, LEARNING_RATE)

    # Train
    trainer.train(
        train_loader,
        val_loader,
        epochs=EPOCHS,
        reg_weight=REG_WEIGHT,
        cls_weight=CLS_WEIGHT
    )

    # Plot history
    trainer.plot_history()

    print(f"\n{'=' * 70}")
    print("Training Complete!")
    print(f"{'=' * 70}")
    print("\nGenerated files:")
    print("  - oximetry_model_trained.pth  (Final model)")
    print("  - oximetry_model_best.pth     (Best validation model)")
    print("  - training_history.png        (Loss curves)")
    print("\nYou can now run model_evaluation.py to test the model!")


if __name__ == "__main__":
    main()