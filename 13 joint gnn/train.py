import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from torch_geometric.data import Data
from tqdm import tqdm
from pathlib import Path
from torch_geometric.nn import RGCNConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import json
from sklearn.metrics import accuracy_score, f1_score

# Configuration
GRAPH_DATA_DIR = Path("graph_data_processed_for_joint_prediction")
NUM_FOLDS = 10
SEED = 42
EPOCHS = 20
BATCH_SIZE = 4
EARLY_STOPPING_PATIENCE = 1
# WEIGHT_DECAY = 1e-3  # Increased from 1e-4
LEARNING_RATE = 1e-5  # Reduced from 1e-3
DROPOUT_RATE = 0.5  # Increased from 0.5
EDGE_DROPOUT_RATE = 0.5  # New regularization

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_graph_data():
    """Load all preprocessed graph data files"""
    all_files = sorted([f for f in GRAPH_DATA_DIR.glob("*.pt") if f.is_file()])
    assert len(all_files) == 40, f"Expected 40 files, found {len(all_files)}"
    return [torch.load(f, weights_only=False) for f in all_files]  


def calculate_balanced_weights(data_list, device='cpu', eps=1e-8):
    """Add smoothing to prevent zero counts"""
    edge_counts = torch.ones(3, device=device) * eps
    node_counts = torch.ones(2, device=device) * eps
    
    for data in data_list:
        edge_counts += torch.bincount(data.edge_type.to(device), minlength=3)
        node_counts += torch.bincount(data.y.to(device), minlength=2)
        
    edge_weights = edge_counts.sum() / (3 * edge_counts)
    node_weights = node_counts.sum() / (2 * node_counts)
    
    return edge_weights, node_weights




def plot_learning_curves(train_losses, val_losses, output_path, early_stop_epoch=None):
    """Create training/validation loss curves with specified style"""
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-', color='b')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='--', color='orange')
    
    # Add this section for early stopping marker
    if early_stop_epoch is not None:
        plt.axvline(x=early_stop_epoch, color='red', linestyle=':', label=f'Early Stop (Epoch {early_stop_epoch})')
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()


def plot_relation_confusion_matrix(y_true, y_pred, output_path):
    """Create 3-class confusion matrix for relation classification"""
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    cell_labels = np.array([
        [f"Support→Support\n{cm[0,0]}", f"Support→Attack\n{cm[0,1]}", f"Support→NoRel\n{cm[0,2]}"],
        [f"Attack→Support\n{cm[1,0]}", f"Attack→Attack\n{cm[1,1]}", f"Attack→NoRel\n{cm[1,2]}"],
        [f"NoRel→Support\n{cm[2,0]}", f"NoRel→Attack\n{cm[2,1]}", f"NoRel→NoRel\n{cm[2,2]}"]
    ])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=cell_labels, fmt='', cmap="Blues",
                xticklabels=["Pred Support", "Pred Attack", "Pred No-Rel"],
                yticklabels=["Actual Support", "Actual Attack", "Actual No-Rel"])
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix for Relation Classification')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_classification_report(y_true, y_pred, output_path):
    """Save classification report in both JSON and text formats"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # JSON report
    report = classification_report(y_true, y_pred, labels=[0,1,2], 
                                  target_names=["Support", "Attack", "No Relation"],
                                  output_dict=True, zero_division=0)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Text report
    txt_path = output_path.with_suffix('.txt')  # Correct way to change extension
    txt_report = classification_report(y_true, y_pred, labels=[0,1,2],
                                      target_names=["Support", "Attack", "No Relation"],
                                      zero_division=0)
    with open(txt_path, 'w') as f:
        f.write(txt_report)

def save_results(true, pred, out_dir):
    """Save evaluation metrics and reports in specified format"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(true, pred),
        "precision": precision_score(true, pred, average="macro"),
        "recall": recall_score(true, pred, average="macro"),
        "f1": f1_score(true, pred, average="macro")
    }
    
    # Save metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save classification report
    report = classification_report(
        true, pred,
        target_names=["Support", "Attack", "No Relation"],
        digits=4
    )
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    # Plot confusion matrix
    plot_relation_confusion_matrix(true, pred, out_dir / "confusion_matrix.png")

def save_joint_results(edge_data, node_data, out_dir):
    e_true, e_pred = edge_data
    n_true, n_pred = node_data
    
    # Create subdirectories
    edge_dir = out_dir / "edges"
    node_dir = out_dir / "nodes"
    edge_dir.mkdir(parents=True, exist_ok=True)
    node_dir.mkdir(parents=True, exist_ok=True)

    # Save edge reports
    _save_task_results(
        e_true, e_pred,
        target_names=["Support", "Attack", "No Relation"],
        labels=[0, 1, 2],
        output_dir=edge_dir
    )
    
    # Save node reports
    _save_task_results(
        n_true, n_pred,
        target_names=["Premise", "Conclusion"],
        labels=[0, 1],
        output_dir=node_dir
    )

def _save_task_results(true, pred, target_names, labels, output_dir):
    """Helper function to save results for a single task"""
    # Classification report
    report = classification_report(
        true, pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    # Save JSON report
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)
        
    # Save text report
    txt_report = classification_report(
        true, pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0
    )
    with open(output_dir / "report.txt", "w") as f:
        f.write(txt_report)
    
    # Confusion matrix
    _plot_confusion_matrix(
        true, pred,
        labels=labels,
        target_names=target_names,
        output_path=output_dir / "confusion_matrix.png"
    )

def _plot_confusion_matrix(y_true, y_pred, labels, target_names, output_path):
    """Generic confusion matrix plotter"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



# Add this class before your existing functions
class EarlyStopping:
    def __init__(self, patience=2, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=1.0, reduction='mean'): 
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class EnhancedLegalRGCN(torch.nn.Module):
    def __init__(self, in_channels=770, hidden_channels=64, num_relations=3, dropout=DROPOUT_RATE):
        super().__init__()
        # Remove BatchNorm due to small batch sizes
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        
        # Add LayerNorm for stability
        self.norm = torch.nn.LayerNorm(hidden_channels)
        
        # Classifiers with weight initialization
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 3)
        )
        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels//2, 2)
        )
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x, edge_index, edge_type):
        # Shared Encoding with residual connections
        x1 = F.relu(self.conv1(x, edge_index, edge_type))
        x2 = F.relu(self.conv2(x1, edge_index, edge_type))
        x3 = self.norm(self.conv3(x2, edge_index, edge_type))
        
        # Edge Classification with log softmax for stability
        row, col = edge_index
        edge_features = torch.cat([x3[row], x3[col]], dim=-1)
        edge_out = F.log_softmax(self.edge_classifier(edge_features), dim=-1)
        
        # Node Classification
        node_out = F.log_softmax(self.node_classifier(x3), dim=-1)
        
        return edge_out, node_out


class MultiTaskLoss(torch.nn.Module):
    def __init__(self, edge_weights, node_weights, alpha=0.6):
        super().__init__()
        # Use NLLLoss with log softmax outputs
        self.edge_criterion = torch.nn.NLLLoss(weight=edge_weights)
        self.node_criterion = torch.nn.NLLLoss(weight=node_weights)
        self.alpha = alpha

    def forward(self, edge_pred, edge_true, node_pred, node_true):
        edge_loss = self.edge_criterion(edge_pred, edge_true)
        node_loss = self.node_criterion(node_pred, node_true)
        return self.alpha * edge_loss + (1 - self.alpha) * node_loss



def train_epoch(model, train_data, optimizer, criterion):
    model.train()
    total_loss = 0
    valid_batches = 0
    
    for data in train_data:
        data = data.to(device)

        if data.edge_index.size(1) == 0:
            continue
            
        # Safer edge dropout
        if model.training and EDGE_DROPOUT_RATE > 0:
            num_edges = data.edge_index.size(1)
            mask = torch.rand(num_edges, device=device) > EDGE_DROPOUT_RATE
            if mask.sum() == 0:  # Keep at least one edge
                mask[0] = True
            data.edge_index = data.edge_index[:, mask]
            data.edge_type = data.edge_type[mask]
            
        data = data.to(device)
        optimizer.zero_grad()
        
        edge_pred, node_pred = model(data.x, data.edge_index, data.edge_type)
        loss = criterion(edge_pred, data.edge_type, node_pred, data.y)
        
        # Check for valid loss
        if torch.isnan(loss):
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Adjusted clipping
        optimizer.step()
        
        total_loss += loss.item()
        valid_batches += 1
        
    return total_loss / max(1, valid_batches)




def validate(model, val_data, criterion):
    model.eval()
    edge_preds, edge_true = [], []
    node_preds, node_true = [], []
    total_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for data in val_data:
            data = data.to(device)
            edge_out, node_out = model(data.x, data.edge_index, data.edge_type)
            
            loss = criterion(edge_out, data.edge_type, node_out, data.y)
            total_loss += loss.item()
            valid_batches += 1
            
            # Collect predictions
            edge_preds.extend(edge_out.argmax(dim=1).cpu().numpy())
            edge_true.extend(data.edge_type.cpu().numpy())
            node_preds.extend(node_out.argmax(dim=1).cpu().numpy())
            node_true.extend(data.y.cpu().numpy())
    
    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
    
    edge_f1 = f1_score(edge_true, edge_preds, average='macro')
    node_f1 = f1_score(node_true, node_preds, average='macro')
    combined_f1 = 0.7*edge_f1 + 0.3*node_f1  # Use same weighting as loss

    return avg_loss, combined_f1, (edge_true, edge_preds), (node_true, node_preds)


def run_cross_validation():
    # Initialize
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    all_data = load_graph_data()
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_data)):  # Direct train/val split
        print(f"\n=== Fold {fold+1}/{NUM_FOLDS} ===")
        fold_dir = Path(f"fold_{fold+1}")
        fold_dir.mkdir(exist_ok=True)

        # Split data - no separate test set
        train_data = [all_data[i] for i in train_idx]
        val_data = [all_data[i] for i in val_idx]  # Validation = test in this setup

        # Initialize model
        model = EnhancedLegalRGCN().to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=LEARNING_RATE, 
            # weight_decay=WEIGHT_DECAY
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        edge_weights, node_weights = calculate_balanced_weights(train_data, device=device)
        criterion = MultiTaskLoss(
            edge_weights=edge_weights.to(device),
            node_weights=node_weights.to(device),
            alpha=0.6
        )

        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001)
        
        train_losses = []
        val_losses = []
        early_stop_epoch = None
        
        # Training loop
        best_f1 = -np.inf
        for epoch in range(1, EPOCHS+1):
            train_loss = train_epoch(model, train_data, optimizer, criterion)
            val_loss, val_f1, edge_data, node_data = validate(model, val_data, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_f1)

            if val_f1 > best_f1 or epoch == 1:
                best_f1 = val_f1
                torch.save(model.state_dict(), f'fold_{fold+1}_best_model.pt')
                print(f"Saved new best model with Val F1: {best_f1:.4f}")

            print(f'Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}')

            if early_stopping(val_loss, model):
                print(f'Early stopping triggered at epoch {epoch}')
                early_stop_epoch = epoch
                break

        plot_learning_curves(train_losses, val_losses, fold_dir/"learning_curve.png", early_stop_epoch)

        # Final validation evaluation with best model
        model.load_state_dict(torch.load(f'fold_{fold+1}_best_model.pt'))
        _, final_f1, edge_data, node_data = validate(model, val_data, criterion)
        fold_metrics.append(final_f1)
        
        # Save validation results
        print(f"\n=== Fold {fold+1} Final Validation Metrics ===")
        print(f"Edge F1: {f1_score(edge_data[0], edge_data[1], average='macro'):.4f}")
        print(f"Node F1: {f1_score(node_data[0], node_data[1], average='macro'):.4f}")
        save_joint_results(edge_data, node_data, fold_dir)

    # Final cross-validation report
    print("\n=== Cross-Validation Results ===")
    print(f"Average Validation F1 across folds: {np.mean(fold_metrics):.4f} ± {np.std(fold_metrics):.4f}")
    plt.figure(figsize=(10,6))
    plt.plot(range(1, NUM_FOLDS+1), fold_metrics, marker='o')
    plt.title("Cross-Validation Performance per Fold")
    plt.xlabel("Fold Number")
    plt.ylabel("Validation F1 Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig('cross_val_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_cross_validation()
