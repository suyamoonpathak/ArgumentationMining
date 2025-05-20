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

# Configuration
GRAPH_DATA_DIR = Path("graph_data_4")
NUM_FOLDS = 10
SEED = 42
EPOCHS = 10
BATCH_SIZE = 4

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_graph_data():
    """Load all preprocessed graph data files"""
    all_files = sorted([f for f in GRAPH_DATA_DIR.glob("*.pt") if f.is_file()])
    assert len(all_files) == 40, f"Expected 40 files, found {len(all_files)}"
    return [torch.load(f, weights_only=False) for f in all_files]  # Only use if you trust the data files


def calculate_balanced_class_weights(data_list):
    """Calculate class weights for imbalance correction"""
    total_support = sum((data.edge_type == 0).sum().item() for data in data_list)
    total_attack = sum((data.edge_type == 1).sum().item() for data in data_list)
    total_no_relation = sum((data.edge_type == 2).sum().item() for data in data_list)
    
    total_samples = total_support + total_attack + total_no_relation
    support_w = total_samples / (3 * total_support) if total_support > 0 else 0
    attack_w = total_samples / (3 * total_attack) if total_attack > 0 else 0
    no_rel_w = total_samples / (3 * total_no_relation) if total_no_relation > 0 else 0
    
    class_weights = torch.tensor([support_w, attack_w, no_rel_w], dtype=torch.float)
    print(f"Class weights - Support: {support_w:.2f}, Attack: {attack_w:.2f}, No Relation: {no_rel_w:.2f}")
    return class_weights

def plot_learning_curves(train_losses, val_losses, output_path):
    """Create training/validation loss curves with specified style"""
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-', color='b')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='--', color='orange')
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



class FocalLoss(torch.nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

class EnhancedLegalRGCN(torch.nn.Module):
    def __init__(self, in_channels=770, hidden_channels=64, num_relations=3, dropout=0.5):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels//2, 3)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        x1 = F.relu(self.conv1(x, edge_index, edge_type))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        x2 = F.relu(self.conv2(x1, edge_index, edge_type))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        x3 = self.conv3(x2, edge_index, edge_type) + x1  # Skip connection
        
        row, col = edge_index
        edge_features = torch.cat([x3[row], x3[col]], dim=-1)
        return self.classifier(edge_features)

def train_epoch(model, train_data, optimizer, criterion):
    model.train()
    total_loss = 0
    valid_batches = 0
    
    for data in train_data:
        if data.edge_index.size(1) == 0:  # Skip graphs with no edges
            continue
            
        max_idx = max(data.edge_index[0].max().item(), data.edge_index[1].max().item())
        if max_idx >= data.x.size(0):
            continue
            
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index, data.edge_type)
        loss = criterion(out, data.edge_type)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        valid_batches += 1
        
    return total_loss / max(1, valid_batches)

def validate(model, val_data, criterion):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for data in val_data:
            if data.edge_index.size(1) == 0:
                continue
                
            max_idx = max(data.edge_index[0].max().item(), data.edge_index[1].max().item())
            if max_idx >= data.x.size(0):
                continue
                
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_type)
            
            loss = criterion(out, data.edge_type)
            total_loss += loss.item()
            valid_batches += 1
            
            pred = out.argmax(dim=1).cpu().numpy()
            true = data.edge_type.cpu().numpy()
            
            all_preds.extend(pred)
            all_true.extend(true)
    
    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average=None, labels=[0,1,2], zero_division=0
    )
    return avg_loss, f1.mean()

def run_cross_validation():
    # Initialize
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    all_data = load_graph_data()
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
        print(f"\n=== Fold {fold+1}/{NUM_FOLDS} ===")
        fold_dir = Path(f"fold_{fold+1}")
        fold_dir.mkdir(exist_ok=True)

        
        # Split data
        train_val = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]
        
        # Create validation split
        val_size = int(0.2 * len(train_val))
        train_data, val_data = train_val[val_size:], train_val[:val_size]

        # Initialize model
        model = EnhancedLegalRGCN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        class_weights = calculate_balanced_class_weights(train_data)
        criterion = FocalLoss(weight=class_weights.to(device), gamma=2.0)

        train_losses = []
        val_losses = []
        
        # Training loop
        best_f1 = 0
        for epoch in range(1, EPOCHS+1):
            train_loss = train_epoch(model, train_data, optimizer, criterion)
            val_loss, val_f1 = validate(model, val_data, criterion)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_f1)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(model.state_dict(), f'fold_{fold+1}_best_model.pt')

            print(f'Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val F1: {val_f1:.4f}')

        plot_learning_curves(train_losses, val_losses, fold_dir/"learning_curve.png")

        
        # Test evaluation
        all_true, all_preds = [], []
        model.load_state_dict(torch.load(f'fold_{fold+1}_best_model.pt'))
        for data in test_data:
            data = data.to(device)
            with torch.no_grad():
                out = model(data.x, data.edge_index, data.edge_type)
                preds = out.argmax(dim=1).cpu().numpy()
                true = data.edge_type.cpu().numpy()
                
                all_preds.extend(preds)
                all_true.extend(true)
        
        # Save reports
        save_results(all_true, all_preds, fold_dir)

    # Final report
    print("\n=== Cross-Validation Results ===")
    print(f"Average F1 across folds: {np.mean(fold_metrics):.4f} ± {np.std(fold_metrics):.4f}")
    plt.figure(figsize=(10,6))
    plt.plot(range(1, NUM_FOLDS+1), fold_metrics, marker='o')
    plt.title("Cross-Validation Performance per Fold")
    plt.xlabel("Fold Number")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig('cross_val_results.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    run_cross_validation()
