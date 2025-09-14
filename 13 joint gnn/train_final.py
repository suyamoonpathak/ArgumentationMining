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
from sklearn.metrics import f1_score

# Configuration
GRAPH_DATA_DIR = Path("roberta_processed_graph_data_for_joint_prediction")
NUM_FOLDS = 10
SEED = 42
EPOCHS = 10
BATCH_SIZE = 4
EARLY_STOPPING_PATIENCE = 2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_graph_data():
    """Load all preprocessed graph data files"""
    all_files = sorted([f for f in GRAPH_DATA_DIR.glob("*.pt") if f.is_file()])
    assert len(all_files) == 40, f"Expected 40 files, found {len(all_files)}"
    return [torch.load(f, weights_only=False) for f in all_files]  


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

def save_joint_results(node_true, node_pred, edge_true, edge_pred, out_dir):
    """Save evaluation metrics for both node and edge prediction"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Node classification metrics
    node_metrics = {
        "accuracy": accuracy_score(node_true, node_pred),
        "precision": precision_score(node_true, node_pred, average="macro", zero_division=0),
        "recall": recall_score(node_true, node_pred, average="macro", zero_division=0),
        "f1": f1_score(node_true, node_pred, average="macro", zero_division=0)
    }
    
    # Edge classification metrics
    edge_metrics = {
        "accuracy": accuracy_score(edge_true, edge_pred),
        "precision": precision_score(edge_true, edge_pred, average="macro", zero_division=0),
        "recall": recall_score(edge_true, edge_pred, average="macro", zero_division=0),
        "f1": f1_score(edge_true, edge_pred, average="macro", zero_division=0)
    }
    
    # Combined metrics
    joint_metrics = {
        "node_classification": node_metrics,
        "edge_classification": edge_metrics
    }
    
    # Save metrics
    with open(out_dir / "joint_metrics.json", "w") as f:
        json.dump(joint_metrics, f, indent=2)
    
    # Save node classification report
    node_report = classification_report(
        node_true, node_pred,
        target_names=["Premise", "Conclusion"],
        digits=4
    )
    with open(out_dir / "node_classification_report.txt", "w") as f:
        f.write("=== NODE CLASSIFICATION REPORT ===\n")
        f.write(node_report)
    
    # Save edge classification report
    edge_report = classification_report(
        edge_true, edge_pred,
        target_names=["Support", "Attack", "No Relation"],
        digits=4
    )
    with open(out_dir / "edge_classification_report.txt", "w") as f:
        f.write("=== EDGE CLASSIFICATION REPORT ===\n")
        f.write(edge_report)
    
    # Plot confusion matrices
    plot_node_confusion_matrix(node_true, node_pred, out_dir / "node_confusion_matrix.png")
    plot_relation_confusion_matrix(edge_true, edge_pred, out_dir / "edge_confusion_matrix.png")


def plot_node_confusion_matrix(y_true, y_pred, output_path):
    """Create confusion matrix for node classification (premise/conclusion)"""
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    cell_labels = np.array([
        [f"Prem→Prem\n{cm[0,0]}", f"Prem→Conc\n{cm[0,1]}"],
        [f"Conc→Prem\n{cm[1,0]}", f"Conc→Conc\n{cm[1,1]}"]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=cell_labels, fmt='', cmap="Blues",
                xticklabels=["Pred Premise", "Pred Conclusion"],
                yticklabels=["Actual Premise", "Actual Conclusion"])
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix for Node Classification')
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

class JointPredictionLegalRGCN(torch.nn.Module):
    def __init__(self, in_channels=770, hidden_channels=64, num_relations=3, dropout=0.5):
        super().__init__()
        # RGCN layers for learning node representations
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        
        # Node classification head (premise vs conclusion)
        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels//2, 2)  # 2 classes: premise, conclusion
        )
        
        # Edge classification head (support/attack/no-relation)
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, hidden_channels//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels//2, 3)  # 3 classes: support, attack, no-relation
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        # Forward pass through RGCN layers
        x1 = F.relu(self.conv1(x, edge_index, edge_type))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        
        x2 = F.relu(self.conv2(x1, edge_index, edge_type))
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        
        x3 = self.conv3(x2, edge_index, edge_type) + x1  # Skip connection
        
        # Node predictions
        node_out = self.node_classifier(x3)
        
        # Edge predictions
        row, col = edge_index
        edge_features = torch.cat([x3[row], x3[col]], dim=-1)
        edge_out = self.edge_classifier(edge_features)
        
        return node_out, edge_out

def train_epoch_joint(model, train_data, optimizer, node_criterion, edge_criterion, node_weight=1.0, edge_weight=1.0):
    model.train()
    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    valid_batches = 0
    
    for data in train_data:
        if data.edge_index.size(1) == 0:  # Skip graphs with no edges
            continue
            
        max_idx = max(data.edge_index[0].max().item(), data.edge_index[1].max().item())
        if max_idx >= data.x.size(0):
            continue
            
        data = data.to(device)
        optimizer.zero_grad()
        
        node_out, edge_out = model(data.x, data.edge_index, data.edge_type)
        
        # Calculate losses
        node_loss = node_criterion(node_out, torch.tensor(data.y).to(device))
        edge_loss = edge_criterion(edge_out, data.edge_type)
        
        # Combined loss
        total_loss_batch = node_weight * node_loss + edge_weight * edge_loss
        
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += total_loss_batch.item()
        total_node_loss += node_loss.item()
        total_edge_loss += edge_loss.item()
        valid_batches += 1
        
    avg_total_loss = total_loss / max(1, valid_batches)
    avg_node_loss = total_node_loss / max(1, valid_batches)
    avg_edge_loss = total_edge_loss / max(1, valid_batches)
    
    return avg_total_loss, avg_node_loss, avg_edge_loss


def validate_joint(model, val_data, node_criterion, edge_criterion, node_weight=1.0, edge_weight=1.0):
    model.eval()
    all_node_preds, all_node_true = [], []
    all_edge_preds, all_edge_true = [], []
    total_loss = 0
    total_node_loss = 0
    total_edge_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for data in val_data:
            if data.edge_index.size(1) == 0:
                continue
                
            max_idx = max(data.edge_index[0].max().item(), data.edge_index[1].max().item())
            if max_idx >= data.x.size(0):
                continue
                
            data = data.to(device)
            node_out, edge_out = model(data.x, data.edge_index, data.edge_type)
            
            # Calculate losses
            node_loss = node_criterion(node_out, torch.tensor(data.y).to(device))
            edge_loss = edge_criterion(edge_out, data.edge_type)
            total_loss_batch = node_weight * node_loss + edge_weight * edge_loss
            
            total_loss += total_loss_batch.item()
            total_node_loss += node_loss.item()
            total_edge_loss += edge_loss.item()
            valid_batches += 1
            
            # Collect predictions
            node_pred = node_out.argmax(dim=1).cpu().numpy()
            node_true = np.array(data.y)
            edge_pred = edge_out.argmax(dim=1).cpu().numpy()
            edge_true = data.edge_type.cpu().numpy()
            
            all_node_preds.extend(node_pred)
            all_node_true.extend(node_true)
            all_edge_preds.extend(edge_pred)
            all_edge_true.extend(edge_true)
    
    avg_loss = total_loss / max(1, valid_batches)
    avg_node_loss = total_node_loss / max(1, valid_batches)
    avg_edge_loss = total_edge_loss / max(1, valid_batches)
    
    # Calculate F1 scores
    node_precision, node_recall, node_f1, _ = precision_recall_fscore_support(
        all_node_true, all_node_preds, average='macro', zero_division=0
    )
    edge_precision, edge_recall, edge_f1, _ = precision_recall_fscore_support(
        all_edge_true, all_edge_preds, average='macro', labels=[0,1,2], zero_division=0
    )
    
    return avg_loss, avg_node_loss, avg_edge_loss, node_f1, edge_f1, all_node_true, all_node_preds, all_edge_true, all_edge_preds


def run_joint_cross_validation():
    # Initialize
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    all_data = load_graph_data()
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    
    node_fold_metrics = []
    edge_fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(all_data)):
        print(f"\n=== Fold {fold+1}/{NUM_FOLDS} ===")
        fold_dir = Path(f"joint_fold_{fold+1}")
        fold_dir.mkdir(exist_ok=True)
        
        # Split data
        train_val = [all_data[i] for i in train_idx]
        test_data = [all_data[i] for i in test_idx]
        
        # Create validation split
        val_size = int(0.2 * len(train_val))
        train_data, val_data = train_val[val_size:], train_val[:val_size]

        # Initialize model
        model = JointPredictionLegalRGCN().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Loss functions
        node_criterion = torch.nn.CrossEntropyLoss()  # For node classification
        edge_class_weights = calculate_balanced_class_weights(train_data)
        edge_criterion = FocalLoss(weight=edge_class_weights.to(device), gamma=2.0)
        
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=0.001)
        
        train_losses, val_losses = [], []
        train_node_losses, train_edge_losses = [], []
        val_node_losses, val_edge_losses = [], []
        early_stop_epoch = None
        
        # Training loop
        best_combined_f1 = 0
        node_weight, edge_weight = 0.4, 0.6  # Give more weight to edge prediction
        
        for epoch in range(1, EPOCHS+1):
            train_loss, train_node_loss, train_edge_loss = train_epoch_joint(
                model, train_data, optimizer, node_criterion, edge_criterion, node_weight, edge_weight
            )
            
            val_loss, val_node_loss, val_edge_loss, val_node_f1, val_edge_f1, _, _, _, _ = validate_joint(
                model, val_data, node_criterion, edge_criterion, node_weight, edge_weight
            )
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_node_losses.append(train_node_loss)
            train_edge_losses.append(train_edge_loss)
            val_node_losses.append(val_node_loss)
            val_edge_losses.append(val_edge_loss)
            
            combined_f1 = (val_node_f1 + val_edge_f1) / 2
            
            if combined_f1 > best_combined_f1:
                best_combined_f1 = combined_f1
                torch.save(model.state_dict(), f'joint_fold_{fold+1}_best_model.pt')

            print(f'Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
            print(f'         | Node F1: {val_node_f1:.4f} | Edge F1: {val_edge_f1:.4f} | Combined F1: {combined_f1:.4f}')

            if early_stopping(val_loss, model):
                print(f'Early stopping triggered at epoch {epoch}')
                early_stop_epoch = epoch
                break

        # Plot learning curves
        plot_joint_learning_curves(train_losses, val_losses, train_node_losses, train_edge_losses,
                                 val_node_losses, val_edge_losses, fold_dir/"joint_learning_curve.png", early_stop_epoch)
        
        # Test evaluation
        model.load_state_dict(torch.load(f'joint_fold_{fold+1}_best_model.pt'))
        all_node_true, all_node_preds = [], []
        all_edge_true, all_edge_preds = [], []
        
        for data in test_data:
            data = data.to(device)
            with torch.no_grad():
                node_out, edge_out = model(data.x, data.edge_index, data.edge_type)
                
                node_preds = node_out.argmax(dim=1).cpu().numpy()
                node_true = np.array(data.y)
                edge_preds = edge_out.argmax(dim=1).cpu().numpy()
                edge_true = data.edge_type.cpu().numpy()
                
                all_node_preds.extend(node_preds)
                all_node_true.extend(node_true)
                all_edge_preds.extend(edge_preds)
                all_edge_true.extend(edge_true)
        
        # Calculate test metrics
        node_test_f1 = f1_score(all_node_true, all_node_preds, average='macro', zero_division=0)
        edge_test_f1 = f1_score(all_edge_true, all_edge_preds, average='macro', zero_division=0)
        
        node_fold_metrics.append(node_test_f1)
        edge_fold_metrics.append(edge_test_f1)
        
        # Save results
        save_joint_results(all_node_true, all_node_preds, all_edge_true, all_edge_preds, fold_dir)

    # Final report
    print("\n=== Joint Cross-Validation Results ===")
    print(f"Node Classification - Average F1: {np.mean(node_fold_metrics):.4f} ± {np.std(node_fold_metrics):.4f}")
    print(f"Edge Classification - Average F1: {np.mean(edge_fold_metrics):.4f} ± {np.std(edge_fold_metrics):.4f}")
    
    # Plot results
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(range(1, NUM_FOLDS+1), node_fold_metrics, marker='o', color='blue', label='Node F1')
    plt.title("Node Classification Performance per Fold")
    plt.xlabel("Fold Number")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.grid(True)
    
    plt.subplot(1,2,2)
    plt.plot(range(1, NUM_FOLDS+1), edge_fold_metrics, marker='s', color='red', label='Edge F1')
    plt.title("Edge Classification Performance per Fold")
    plt.xlabel("Fold Number")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('joint_cross_val_results.png', dpi=300)
    plt.show()


def plot_joint_learning_curves(train_losses, val_losses, train_node_losses, train_edge_losses,
                             val_node_losses, val_edge_losses, output_path, early_stop_epoch=None):
    """Create learning curves for joint training"""
    epochs = np.arange(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0,0].plot(epochs, train_losses, label='Training Loss', marker='o', color='b')
    axes[0,0].plot(epochs, val_losses, label='Validation Loss', marker='x', color='orange')
    if early_stop_epoch:
        axes[0,0].axvline(x=early_stop_epoch, color='red', linestyle=':', label=f'Early Stop (Epoch {early_stop_epoch})')
    axes[0,0].set_title("Total Loss")
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Node loss
    axes[0,1].plot(epochs, train_node_losses, label='Training Node Loss', marker='o', color='b')
    axes[0,1].plot(epochs, val_node_losses, label='Validation Node Loss', marker='x', color='orange')
    if early_stop_epoch:
        axes[0,1].axvline(x=early_stop_epoch, color='red', linestyle=':', label=f'Early Stop (Epoch {early_stop_epoch})')
    axes[0,1].set_title("Node Classification Loss")
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Edge loss
    axes[1,0].plot(epochs, train_edge_losses, label='Training Edge Loss', marker='o', color='b')
    axes[1,0].plot(epochs, val_edge_losses, label='Validation Edge Loss', marker='x', color='orange')
    if early_stop_epoch:
        axes[1,0].axvline(x=early_stop_epoch, color='red', linestyle=':', label=f'Early Stop (Epoch {early_stop_epoch})')
    axes[1,0].set_title("Edge Classification Loss")
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Combined view
    axes[1,1].plot(epochs, train_losses, label='Total Training Loss', color='blue', alpha=0.7)
    axes[1,1].plot(epochs, val_losses, label='Total Validation Loss', color='red', alpha=0.7)
    axes[1,1].plot(epochs, train_node_losses, label='Node Training Loss', color='green', alpha=0.5)
    axes[1,1].plot(epochs, train_edge_losses, label='Edge Training Loss', color='purple', alpha=0.5)
    axes[1,1].set_title("All Losses Combined")
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    run_joint_cross_validation()
