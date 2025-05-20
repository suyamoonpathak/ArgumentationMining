# Import additional libraries for visualization and analysis
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
from sklearn.manifold import TSNE
import pandas as pd

# Configuration
PROCESSED_EMBED_DIR = Path("graph_data_processed_roberta")
RAW_EMBED_DIR = Path("graph_data_raw_embeddings")
NUM_FOLDS = 10
SEED = 42
EPOCHS = 10
BATCH_SIZE = 4

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_graph_data(data_dir):
    """Load all preprocessed graph data files"""
    all_files = sorted([f for f in data_dir.glob("*.pt") if f.is_file()])
    assert len(all_files) > 0, f"No files found in {data_dir}"
    return [torch.load(f, weights_only=False) for f in all_files]
    
def compare_basic_stats(raw_data, proc_data):
    """Compare basic statistics of raw and processed embeddings"""
    if not raw_data or not proc_data:
        print("One or both datasets are empty.")
        return
    
    # Get a sample from each dataset
    raw_sample = raw_data[0]
    proc_sample = proc_data[0]
    
    # Print basic info
    print(f"\n=== Basic Embedding Statistics ===")
    print(f"Raw embedding shape: {raw_sample.x.shape}")
    print(f"Processed embedding shape: {proc_sample.x.shape}")
    
    print("\nRaw embedding stats:")
    print(f"  Mean: {raw_sample.x.mean().item():.4f}")
    print(f"  Std: {raw_sample.x.std().item():.4f}")
    print(f"  Min: {raw_sample.x.min().item():.4f}")
    print(f"  Max: {raw_sample.x.max().item():.4f}")
    
    print("\nProcessed embedding stats:")
    print(f"  Mean: {proc_sample.x.mean().item():.4f}")
    print(f"  Std: {proc_sample.x.std().item():.4f}")
    print(f"  Min: {proc_sample.x.min().item():.4f}")
    print(f"  Max: {proc_sample.x.max().item():.4f}")
    
    # Create distribution plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(raw_sample.x.flatten().cpu().numpy(), kde=True, color='blue')
    plt.title("Raw Embedding Distribution")
    plt.xlabel("Value")
    
    plt.subplot(1, 2, 2)
    sns.histplot(proc_sample.x.flatten().cpu().numpy(), kde=True, color='orange')
    plt.title("Processed Embedding Distribution")
    plt.xlabel("Value")
    
    plt.tight_layout()
    plt.savefig("embedding_distributions.png")
    plt.close()

# Create a modified EnhancedLegalRGCN that tracks layer outputs
class LayerTrackingRGCN(torch.nn.Module):
    def __init__(self, in_channels=770, hidden_channels=64, num_relations=3, dropout=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
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
        
        # Storage for layer activations
        self.activations = {}
        
    def get_activations(self):
        return self.activations

    def forward(self, x, edge_index, edge_type):
        # Store input
        self.activations['input'] = x.clone().detach()
        
        # First layer
        x1_pre = self.conv1(x, edge_index, edge_type)
        self.activations['conv1_pre_relu'] = x1_pre.clone().detach()
        
        x1 = F.relu(x1_pre)
        self.activations['conv1_post_relu'] = x1.clone().detach()
        
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        self.activations['conv1_post_dropout'] = x1.clone().detach()

        # Second layer
        x2_pre = self.conv2(x1, edge_index, edge_type)
        self.activations['conv2_pre_relu'] = x2_pre.clone().detach()
        
        x2 = F.relu(x2_pre)
        self.activations['conv2_post_relu'] = x2.clone().detach()
        
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        self.activations['conv2_post_dropout'] = x2.clone().detach()

        # Third layer with skip connection
        x3 = self.conv3(x2, edge_index, edge_type) + x1
        self.activations['conv3_with_skip'] = x3.clone().detach()

        # Edge features
        row, col = edge_index
        edge_features = torch.cat([x3[row], x3[col]], dim=-1)
        self.activations['edge_features'] = edge_features.clone().detach()
        
        # Final output
        output = self.classifier(edge_features)
        self.activations['output'] = output.clone().detach()
        
        return output

def visualize_activations(raw_acts, proc_acts, output_dir):
    """
    Visualize and compare activations from the two models
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure both have same keys
    common_keys = set(raw_acts.keys()).intersection(proc_acts.keys())
    
    # Compare statistics for each layer
    stats = {}
    
    for key in common_keys:
        raw_act = raw_acts[key].cpu()
        proc_act = proc_acts[key].cpu()
        
        # Calculate statistics
        stats[key] = {
            'raw': {
                'mean': raw_act.mean().item(),
                'std': raw_act.std().item(),
                'min': raw_act.min().item(),
                'max': raw_act.max().item(),
                'shape': list(raw_act.shape)
            },
            'processed': {
                'mean': proc_act.mean().item(),
                'std': proc_act.std().item(),
                'min': proc_act.min().item(),
                'max': proc_act.max().item(),
                'shape': list(proc_act.shape)
            }
        }
        
        # If shapes match, calculate distances
        if raw_act.shape == proc_act.shape:
            # Normalized L2 distance
            l2_dist = torch.norm(raw_act - proc_act) / torch.norm(raw_act)
            stats[key]['normalized_l2_distance'] = l2_dist.item()
            
            # Cosine similarity if applicable
            if len(raw_act.shape) > 1 and raw_act.shape[1] > 1:
                raw_flat = raw_act.reshape(raw_act.shape[0], -1)
                proc_flat = proc_act.reshape(proc_act.shape[0], -1)
                
                # Normalize
                raw_norm = F.normalize(raw_flat, p=2, dim=1)
                proc_norm = F.normalize(proc_flat, p=2, dim=1)
                
                # Calculate cosine similarity
                cos_sim = (raw_norm * proc_norm).sum(1).mean()
                stats[key]['cosine_similarity'] = cos_sim.item()
        
        # Create distribution plots
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.histplot(raw_act.flatten().numpy(), kde=True, color='blue')
        plt.title(f"Raw: {key}")
        plt.xlabel("Activation Value")
        
        plt.subplot(1, 2, 2)
        sns.histplot(proc_act.flatten().numpy(), kde=True, color='orange')
        plt.title(f"Processed: {key}")
        plt.xlabel("Activation Value")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{key}_distribution.png")
        plt.close()
    
    # Save statistics
    with open(output_dir / "activation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    # Create summary plot of differences
    if common_keys:
        plt.figure(figsize=(14, 8))
        
        # Plot means
        plt.subplot(2, 2, 1)
        means_raw = [stats[k]['raw']['mean'] for k in common_keys]
        means_proc = [stats[k]['processed']['mean'] for k in common_keys]
        plt.plot(means_raw, 'b-o', label='Raw')
        plt.plot(means_proc, 'r-o', label='Processed')
        plt.xticks(range(len(common_keys)), [k for k in common_keys], rotation=45, ha='right')
        plt.title("Mean Activation Values")
        plt.legend()
        plt.grid(True)
        
        # Plot standard deviations
        plt.subplot(2, 2, 2)
        stds_raw = [stats[k]['raw']['std'] for k in common_keys]
        stds_proc = [stats[k]['processed']['std'] for k in common_keys]
        plt.plot(stds_raw, 'b-o', label='Raw')
        plt.plot(stds_proc, 'r-o', label='Processed')
        plt.xticks(range(len(common_keys)), [k for k in common_keys], rotation=45, ha='right')
        plt.title("Standard Deviation of Activations")
        plt.legend()
        plt.grid(True)
        
        # Plot L2 distances if available
        l2_dists = [stats[k].get('normalized_l2_distance', 0) for k in common_keys]
        if any(l2_dists):
            plt.subplot(2, 2, 3)
            plt.bar(range(len(common_keys)), l2_dists)
            plt.xticks(range(len(common_keys)), [k for k in common_keys], rotation=45, ha='right')
            plt.title("Normalized L2 Distance Between Activations")
            plt.grid(True)
        
        # Plot cosine similarities if available
        cos_sims = [stats[k].get('cosine_similarity', 0) for k in common_keys]
        if any(cos_sims):
            plt.subplot(2, 2, 4)
            plt.bar(range(len(common_keys)), cos_sims)
            plt.xticks(range(len(common_keys)), [k for k in common_keys], rotation=45, ha='right')
            plt.title("Cosine Similarity Between Activations")
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / "activation_comparison_summary.png")
        plt.close()
    
    return stats

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

    avg_loss = total_loss / max(1, valid_batches)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_true, all_preds, average=None, labels=[0,1,2], zero_division=0
    )
    return avg_loss, f1.mean()

def compare_embeddings_and_models():
    # Set random seeds
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Load both datasets
    try:
        raw_data = load_graph_data(RAW_EMBED_DIR)
        proc_data = load_graph_data(PROCESSED_EMBED_DIR)
        
        print(f"Loaded {len(raw_data)} raw embedding graphs and {len(proc_data)} processed embedding graphs")
        
        # Compare basic statistics
        compare_basic_stats(raw_data, proc_data)
        
        # If we have data in both sets, continue with more analysis
        if len(raw_data) > 0 and len(proc_data) > 0:
            # Get dimensions from data
            raw_dim = raw_data[0].x.shape[1]
            proc_dim = proc_data[0].x.shape[1]
            
            print(f"Raw embedding dimension: {raw_dim}")
            print(f"Processed embedding dimension: {proc_dim}")
            
            # Initialize models
            raw_model = LayerTrackingRGCN(in_channels=raw_dim).to(device)
            proc_model = LayerTrackingRGCN(in_channels=proc_dim).to(device)
            
            # Get a sample graph for analysis
            raw_sample = raw_data[0].to(device)
            proc_sample = proc_data[0].to(device)
            
            # Forward pass to collect activations
            with torch.no_grad():
                _ = raw_model(raw_sample.x, raw_sample.edge_index, raw_sample.edge_type)
                raw_acts = raw_model.get_activations()
                
                _ = proc_model(proc_sample.x, proc_sample.edge_index, proc_sample.edge_type)
                proc_acts = proc_model.get_activations()
            
            # Analyze and visualize
            visualize_activations(raw_acts, proc_acts, "activation_comparison")
            
            # Train a model on each dataset for 1 epoch to see training dynamics
            print("\n=== Training for 1 epoch on each dataset ===")
            
            # Split data (simple 80/20 split)
            raw_train_size = int(len(raw_data) * 0.8)
            proc_train_size = int(len(proc_data) * 0.8)
            
            raw_train = raw_data[:raw_train_size]
            raw_val = raw_data[raw_train_size:]
            
            proc_train = proc_data[:proc_train_size]
            proc_val = proc_data[proc_train_size:]
            
            # Setup training for raw embeddings
            raw_optimizer = torch.optim.AdamW(raw_model.parameters(), lr=0.001)
            raw_class_weights = calculate_balanced_class_weights(raw_train)
            raw_criterion = FocalLoss(weight=raw_class_weights.to(device))
            
            # Train for 1 epoch
            print("Training with raw embeddings...")
            raw_train_loss = train_epoch(raw_model, raw_train, raw_optimizer, raw_criterion)
            raw_val_loss, raw_val_f1 = validate(raw_model, raw_val, raw_criterion)
            
            # Setup training for processed embeddings
            proc_optimizer = torch.optim.AdamW(proc_model.parameters(), lr=0.001)
            proc_class_weights = calculate_balanced_class_weights(proc_train)
            proc_criterion = FocalLoss(weight=proc_class_weights.to(device))
            
            # Train for 1 epoch
            print("Training with processed embeddings...")
            proc_train_loss = train_epoch(proc_model, proc_train, proc_optimizer, proc_criterion)
            proc_val_loss, proc_val_f1 = validate(proc_model, proc_val, proc_criterion)
            
            # Compare results
            print("\n=== Initial Training Results ===")
            print(f"Raw Embeddings - Train Loss: {raw_train_loss:.4f}, Val F1: {raw_val_f1:.4f}")
            print(f"Processed Embeddings - Train Loss: {proc_train_loss:.4f}, Val F1: {proc_val_f1:.4f}")
            
            # Get activations after training
            with torch.no_grad():
                _ = raw_model(raw_sample.x, raw_sample.edge_index, raw_sample.edge_type)
                raw_trained_acts = raw_model.get_activations()
                
                _ = proc_model(proc_sample.x, proc_sample.edge_index, proc_sample.edge_type)
                proc_trained_acts = proc_model.get_activations()
            
            # Analyze and visualize post-training
            visualize_activations(raw_trained_acts, proc_trained_acts, "post_training_activation_comparison")
            
    except Exception as e:
        print(f"Error during embedding comparison: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    compare_embeddings_and_models()
