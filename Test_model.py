import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os


# ============================================
# MODEL DEFINITION (Same as your original)
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
# DATASET CLASS (Same as your original)
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

        self.data[ppg_columns] = (
                                         self.data[ppg_columns] - self.data[ppg_columns].mean()
                                 ) / self.data[ppg_columns].std()

        self.ppg_columns = ppg_columns

        if 'SpO2_gt' in self.data.columns:
            self.spo2_column = 'SpO2_gt'
        elif 'sp02_1' in self.data.columns:
            self.spo2_column = 'sp02_1'
        else:
            spo2_cols = [col for col in self.data.columns
                         if 'spo2' in col.lower() or 'sp02' in col.lower()]
            if spo2_cols:
                self.spo2_column = spo2_cols[0]
            else:
                raise ValueError("SpO2 column not found!")

        print(f"Dataset loaded: {len(self.data)} rows, {len(self)} samples")

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
# MODEL EVALUATOR
# ============================================
class ModelEvaluator:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def evaluate(self, dataloader, dataset_name="Dataset"):
        """Evaluate model on a dataset"""
        print(f"\n{'=' * 70}")
        print(f"Evaluating on {dataset_name}")
        print(f"{'=' * 70}")

        all_spo2_pred = []
        all_spo2_true = []
        all_cls_pred = []
        all_cls_true = []

        with torch.no_grad():
            for ppg, reg_target, cls_target in tqdm(dataloader, desc="Evaluating"):
                ppg = ppg.to(self.device)

                reg_output, cls_output = self.model(ppg)

                all_spo2_pred.extend(reg_output.cpu().numpy().flatten())
                all_spo2_true.extend(reg_target.cpu().numpy().flatten())
                all_cls_pred.extend(cls_output.cpu().numpy().flatten())
                all_cls_true.extend(cls_target.cpu().numpy().flatten())

        # Convert to numpy arrays
        spo2_pred = np.array(all_spo2_pred)
        spo2_true = np.array(all_spo2_true)
        cls_pred = np.array(all_cls_pred)
        cls_true = np.array(all_cls_true)

        # Calculate metrics
        results = self._calculate_metrics(
            spo2_pred, spo2_true, cls_pred, cls_true, dataset_name
        )

        return results

    def _calculate_metrics(self, spo2_pred, spo2_true, cls_pred, cls_true, dataset_name):
        """Calculate all performance metrics"""
        results = {}

        # Regression metrics
        mae = np.mean(np.abs(spo2_pred - spo2_true))
        rmse = np.sqrt(np.mean((spo2_pred - spo2_true) ** 2))
        mape = np.mean(np.abs((spo2_true - spo2_pred) / (spo2_true + 1e-8))) * 100

        results['regression'] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }

        print(f"\nRegression Metrics (SpO2 Prediction):")
        print(f"  MAE:  {mae:.3f}%")
        print(f"  RMSE: {rmse:.3f}%")
        print(f"  MAPE: {mape:.3f}%")

        # Classification metrics
        # Check if there are both classes
        if len(np.unique(cls_true)) < 2:
            print(f"\n⚠ Warning: Only one class present in {dataset_name}")
            print(f"  Unique classes: {np.unique(cls_true)}")
            results['classification'] = {
                'AUC-ROC': None,
                'accuracy': None,
                'sensitivity': None,
                'specificity': None
            }
            return results

        # AUC-ROC
        auc_roc = roc_auc_score(cls_true, cls_pred)
        results['auc_roc'] = auc_roc

        # Binary predictions (threshold = 0.5)
        cls_pred_binary = (cls_pred >= 0.5).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(cls_true, cls_pred_binary).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        results['classification'] = {
            'AUC-ROC': auc_roc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'confusion_matrix': {
                'TP': int(tp), 'TN': int(tn),
                'FP': int(fp), 'FN': int(fn)
            }
        }

        print(f"\nClassification Metrics (Hypoxemia Detection):")
        print(f"  AUC-ROC:     {auc_roc:.4f}")
        print(f"  Accuracy:    {accuracy:.4f}")
        print(f"  Sensitivity: {sensitivity:.4f} (Recall)")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  Precision:   {precision:.4f}")
        print(f"  F1-Score:    {f1:.4f}")

        print(f"\nConfusion Matrix:")
        print(f"  TP: {tp:4d}  |  FN: {fn:4d}")
        print(f"  FP: {fp:4d}  |  TN: {tn:4d}")

        # Class distribution
        n_positive = int(np.sum(cls_true))
        n_negative = int(len(cls_true) - n_positive)
        print(f"\nClass Distribution:")
        print(f"  Hypoxemia (SpO2<90): {n_positive} ({n_positive / len(cls_true) * 100:.1f}%)")
        print(f"  Normal (SpO2≥90):    {n_negative} ({n_negative / len(cls_true) * 100:.1f}%)")

        # Store predictions for plotting
        results['predictions'] = {
            'spo2_pred': spo2_pred,
            'spo2_true': spo2_true,
            'cls_pred': cls_pred,
            'cls_true': cls_true
        }

        return results

    def plot_results(self, train_results, test_results=None, save_path='evaluation_plots.png'):
        """Plot comprehensive evaluation results"""
        n_plots = 4 if test_results else 2
        fig = plt.figure(figsize=(15, 10))

        # Plot 1: ROC Curve for Training
        ax1 = plt.subplot(2, 2, 1)
        self._plot_roc_curve(train_results, 'Training Set', ax1, 'blue')

        # Plot 2: SpO2 Scatter for Training
        ax2 = plt.subplot(2, 2, 2)
        self._plot_spo2_scatter(train_results, 'Training Set', ax2, 'blue')

        if test_results:
            # Plot 3: ROC Curve for Test
            ax3 = plt.subplot(2, 2, 3)
            self._plot_roc_curve(test_results, 'Test Set', ax3, 'red')

            # Plot 4: SpO2 Scatter for Test
            ax4 = plt.subplot(2, 2, 4)
            self._plot_spo2_scatter(test_results, 'Test Set', ax4, 'red')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plots saved to: {save_path}")
        plt.close()

    def _plot_roc_curve(self, results, dataset_name, ax, color):
        """Plot ROC curve"""
        if results['classification']['AUC-ROC'] is None:
            ax.text(0.5, 0.5, 'Not enough classes',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'ROC Curve - {dataset_name}')
            return

        cls_pred = results['predictions']['cls_pred']
        cls_true = results['predictions']['cls_true']

        fpr, tpr, _ = roc_curve(cls_true, cls_pred)
        auc = results['classification']['AUC-ROC']

        ax.plot(fpr, tpr, color=color, lw=2, label=f'ROC (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {dataset_name}')
        ax.legend()
        ax.grid(alpha=0.3)

    def _plot_spo2_scatter(self, results, dataset_name, ax, color):
        """Plot SpO2 prediction scatter"""
        spo2_pred = results['predictions']['spo2_pred']
        spo2_true = results['predictions']['spo2_true']

        ax.scatter(spo2_true, spo2_pred, alpha=0.3, s=10, color=color)
        ax.plot([70, 100], [70, 100], 'k--', lw=2, label='Perfect Prediction')

        mae = results['regression']['MAE']
        rmse = results['regression']['RMSE']

        ax.set_xlabel('True SpO2 (%)')
        ax.set_ylabel('Predicted SpO2 (%)')
        ax.set_title(f'SpO2 Prediction - {dataset_name}')
        ax.legend([f'MAE={mae:.2f}%, RMSE={rmse:.2f}%'])
        ax.grid(alpha=0.3)
        ax.set_xlim([70, 100])
        ax.set_ylim([70, 100])


# ============================================
# MAIN TESTING SCRIPT
# ============================================
def main():
    print("=" * 70)
    print("MODEL EVALUATION - AUC-ROC and Performance Metrics")
    print("=" * 70)

    # Configuration
    MODEL_PATH = 'oximetry_model_trained.pth'
    TRAIN_PPG = '100002.csv'
    TRAIN_GT = '100002 (1).csv'
    TEST_PPG = 'test_ppg.csv'  # Change to your test file
    TEST_GT = 'test_gt.csv'  # Change to your test file
    WINDOW_SIZE = 300
    BATCH_SIZE = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load model
    print(f"\nLoading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"✗ Error: Model file not found!")
        return

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model = HybridOximetryModel(input_channels=3, window_size=WINDOW_SIZE)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded successfully!")

    # Create evaluator
    evaluator = ModelEvaluator(model, device)

    # Load training dataset
    print(f"\n{'=' * 70}")
    print("Loading Training Dataset")
    print(f"{'=' * 70}")
    try:
        train_dataset = OximetryDataset(TRAIN_PPG, TRAIN_GT, WINDOW_SIZE)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=0)
    except Exception as e:
        print(f"✗ Error loading training dataset: {e}")
        return

    # Evaluate on training set
    train_results = evaluator.evaluate(train_loader, "Training Set")

    # Load and evaluate test dataset (if available)
    test_results = None
    if os.path.exists(TEST_PPG) and os.path.exists(TEST_GT):
        print(f"\n{'=' * 70}")
        print("Loading Test Dataset")
        print(f"{'=' * 70}")
        try:
            test_dataset = OximetryDataset(TEST_PPG, TEST_GT, WINDOW_SIZE)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                     shuffle=False, num_workers=0)
            test_results = evaluator.evaluate(test_loader, "Test Set")
        except Exception as e:
            print(f"⚠ Warning: Could not load test dataset: {e}")
    else:
        print(f"\n⚠ Test dataset not found. Evaluating training set only.")
        print(f"  Expected: {TEST_PPG} and {TEST_GT}")

    # Plot results
    evaluator.plot_results(train_results, test_results)

    # Summary
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nTraining Set:")
    print(f"  AUC-ROC: {train_results['classification']['AUC-ROC']:.4f}")
    print(f"  MAE:     {train_results['regression']['MAE']:.3f}%")

    if test_results:
        print(f"\nTest Set:")
        print(f"  AUC-ROC: {test_results['classification']['AUC-ROC']:.4f}")
        print(f"  MAE:     {test_results['regression']['MAE']:.3f}%")

    print(f"\n{'=' * 70}")
    print("Evaluation Complete!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()