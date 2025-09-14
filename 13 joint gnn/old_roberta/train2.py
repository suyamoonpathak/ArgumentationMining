import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
GRAPH_DATA_DIR = Path("graph_data_processed_for_joint_prediction")
NUM_FOLDS = 10
SEED = 42
EPOCHS = 50
LEARNING_RATE = 15e-6
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EDGE_TYPE_MAP = {0: "Support", 1: "Attack", 2: "No Relation"}
NODE_TYPE_MAP = {0: "Premise", 1: "Conclusion"}

# Helper Functions (from previous code)
def load_graph_data(graph_data_dir):
    """Load graph data with secure deserialization."""
    files = sorted([f for f in graph_data_dir.glob("*.pt") if f.is_file()])
    if not files:
        raise FileNotFoundError(f"No graph data files found in {graph_data_dir}")
    from torch_geometric.data import Data
    torch.serialization.add_safe_globals([Data])
    return [torch.load(f, weights_only=False) for f in files]

def calculate_graph_counts(data_list):
    """Calculate edge and node type counts for each graph."""
    edge_counts, node_counts = [], []
    for data in data_list:
        edge_counts.append(torch.bincount(data.edge_type, minlength=len(EDGE_TYPE_MAP)).numpy())
        node_counts.append(torch.bincount(data.y, minlength=len(NODE_TYPE_MAP)).numpy())
    return np.array(edge_counts), np.array(node_counts)

def create_balanced_splits(data_list, num_folds, seed):
    """Create custom splits to balance edge and node types."""
    np.random.seed(seed)
    edge_counts, node_counts = calculate_graph_counts(data_list)

    total_edge_counts = edge_counts.sum(axis=0)
    total_node_counts = node_counts.sum(axis=0)

    folds = [[] for _ in range(num_folds)]
    fold_edge_counts = np.zeros((num_folds, len(EDGE_TYPE_MAP)))
    fold_node_counts = np.zeros((num_folds, len(NODE_TYPE_MAP)))

    indices = np.arange(len(data_list))
    np.random.shuffle(indices)

    for idx in indices:
        graph_edges, graph_nodes = edge_counts[idx], node_counts[idx]
        scores = []
        for fold in range(num_folds):
            updated_edges = fold_edge_counts[fold] + graph_edges
            updated_nodes = fold_node_counts[fold] + graph_nodes
            edge_imbalance = np.std(updated_edges / total_edge_counts)
            node_imbalance = np.std(updated_nodes / total_node_counts)
            scores.append(edge_imbalance + node_imbalance)
        best_fold = np.argmin(scores)
        folds[best_fold].append(idx)
        fold_edge_counts[best_fold] += graph_edges
        fold_node_counts[best_fold] += graph_nodes

    return folds
class MultiTaskLoss(torch.nn.Module):
    def __init__(self, edge_weight=None, node_weight=None, alpha=0.6):
        super().__init__()
        self.edge_criterion = torch.nn.CrossEntropyLoss(weight=edge_weight)
        self.node_criterion = torch.nn.CrossEntropyLoss(weight=node_weight)
        self.alpha = alpha  # Weight for edge vs. node loss

    def forward(self, edge_pred, edge_true, node_pred, node_true):
        edge_loss = self.edge_criterion(edge_pred, edge_true)
        node_loss = self.node_criterion(node_pred, node_true)
        return self.alpha * edge_loss + (1 - self.alpha) * node_loss

# RGCN Model
class EnhancedLegalRGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_relations, dropout=DROPOUT_RATE):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations)
        self.edge_classifier = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, len(EDGE_TYPE_MAP))
        )
        self.node_classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels // 2, len(NODE_TYPE_MAP))
        )

    def forward(self, x, edge_index, edge_type):
        x1 = F.relu(self.conv1(x, edge_index, edge_type))
        x2 = F.relu(self.conv2(x1, edge_index, edge_type))
        x3 = self.conv3(x2, edge_index, edge_type)

        row, col = edge_index
        edge_features = torch.cat([x3[row], x3[col]], dim=-1)
        edge_out = self.edge_classifier(edge_features)
        node_out = self.node_classifier(x3)

        return edge_out, node_out

# Training and Evaluation
def train_epoch(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to(device)
        optimizer.zero_grad()
        edge_pred, node_pred = model(data.x, data.edge_index, data.edge_type)
        loss = criterion(edge_pred, data.edge_type, node_pred, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)


def validate(model, data_loader, criterion):
    model.eval()
    edge_true, edge_pred = [], []
    node_true, node_pred = [], []
    total_loss = 0

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            edge_out, node_out = model(data.x, data.edge_index, data.edge_type)
            loss = criterion(edge_out, data.edge_type, node_out, data.y)
            total_loss += loss.item()

            edge_true.extend(data.edge_type.cpu().numpy())
            edge_pred.extend(edge_out.argmax(dim=1).cpu().numpy())
            node_true.extend(data.y.cpu().numpy())
            node_pred.extend(node_out.argmax(dim=1).cpu().numpy())

    edge_f1 = f1_score(edge_true, edge_pred, average="macro")
    node_f1 = f1_score(node_true, node_pred, average="macro")
    return total_loss / len(data_loader), edge_f1, node_f1

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

def run_joint_rgcn_cross_validation():
    data_list = load_graph_data(GRAPH_DATA_DIR)
    folds = create_balanced_splits(data_list, NUM_FOLDS, SEED)
    all_f1_scores = []
    output_dir = Path("results")

    for fold_idx, val_indices in enumerate(folds):
        print(f"=== Fold {fold_idx + 1}/{NUM_FOLDS} ===")
        train_indices = [i for i in range(len(data_list)) if i not in val_indices]
        train_data = [data_list[idx] for idx in train_indices]
        val_data = [data_list[idx] for idx in val_indices]

        # Initialize Model
        model = EnhancedLegalRGCN(in_channels=770, hidden_channels=32, num_relations=3).to(device)
        

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        # Compute class weights for balanced training
        edge_weight = calculate_class_weights(train_data, "edge_type", len(EDGE_TYPE_MAP))
        node_weight = calculate_class_weights(train_data, "y", len(NODE_TYPE_MAP))
        criterion = MultiTaskLoss(edge_weight=edge_weight, node_weight=node_weight)

        best_val_loss = float("inf")
        early_stop_counter = 0
        max_patience = 5
        train_losses, val_losses = [], []
        edge_true_all, edge_pred_all = [], []
        node_true_all, node_pred_all = [], []

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_data, optimizer, criterion)
            val_loss, edge_f1, node_f1 = validate(model, val_data, criterion)
            scheduler.step(val_loss)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            print(f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Edge F1: {edge_f1:.4f} | Node F1: {node_f1:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= max_patience:
                    print("Early stopping triggered.")
                    break

        # Collect predictions for evaluation
        for data in val_data:
            data = data.to(device)
            edge_out, node_out = model(data.x, data.edge_index, data.edge_type)
            edge_true_all.extend(data.edge_type.cpu().numpy())
            edge_pred_all.extend(edge_out.argmax(dim=1).cpu().numpy())
            node_true_all.extend(data.y.cpu().numpy())
            node_pred_all.extend(node_out.argmax(dim=1).cpu().numpy())

        # Plot and save learning curves
        fold_dir = output_dir / f"fold_{fold_idx + 1}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        plot_learning_curves(train_losses, val_losses, fold_dir / "learning_curve.png")

        # Save confusion matrix and classification reports
        save_joint_results(
            edge_data=(edge_true_all, edge_pred_all),
            node_data=(node_true_all, node_pred_all),
            out_dir=fold_dir
        )

        # Save F1 scores for fold
        all_f1_scores.append((f1_score(edge_true_all, edge_pred_all, average="macro"),
                              f1_score(node_true_all, node_pred_all, average="macro")))

    avg_edge_f1 = np.mean([score[0] for score in all_f1_scores])
    avg_node_f1 = np.mean([score[1] for score in all_f1_scores])
    print(f"\nFinal Average Edge F1: {avg_edge_f1:.4f}")
    print(f"Final Average Node F1: {avg_node_f1:.4f}")

def calculate_class_weights(data_list, attr, num_classes):
    counts = torch.zeros(num_classes, device=device)  # Ensure counts is on the correct device
    for data in data_list:
        attr_tensor = getattr(data, attr).to(device)  # Move the attribute tensor to the same device
        counts += torch.bincount(attr_tensor, minlength=num_classes)
    weights = 1.0 / (counts + 1e-5)  # Avoid division by zero
    return weights



if __name__ == "__main__":
    run_joint_rgcn_cross_validation()
