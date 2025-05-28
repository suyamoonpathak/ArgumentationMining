import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

# Configuration - Change this path to test different models
MODEL_PATH = "./RoBERTa_best_modelfold1"  # Relative path from script.py location
DATA_PATH = "../../data"  # Path to data folder
RESULTS_PATH = "./results"  # Where to save results

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, label_mapping, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.label_mapping = label_mapping
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Convert string label to numeric using label mapping
        numeric_label = self.label_mapping[label]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(numeric_label, dtype=torch.long)
        }

def load_test_data(data_path):
    """Load all CSV files from the data folder and create label mapping"""
    csv_files = glob.glob(os.path.join(data_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    all_texts = []
    all_labels = []
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        all_texts.extend(df['text'].tolist())
        all_labels.extend(df['label'].tolist())
    
    # Create label mapping from string labels to numeric
    unique_labels = list(set(all_labels))
    unique_labels.sort()  # Sort for consistency
    
    # Create mapping: typically 'premise' -> 1, 'conclusion' -> 0
    if 'premise' in unique_labels and 'conclusion' in unique_labels:
        label_mapping = {'conclusion': 0, 'premise': 1}
    else:
        # Fallback: create mapping based on sorted unique labels
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f"Label mapping: {label_mapping}")
    print(f"Label distribution:")
    label_counts = pd.Series(all_labels).value_counts()
    for label, count in label_counts.items():
        print(f"  {label}: {count}")
    
    return all_texts, all_labels, label_mapping

def plot_confusion_matrix(y_true, y_pred, output_path, label_mapping):
    """Plot confusion matrix in the specified style"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Determine if we're dealing with binary classification
    if cm.shape == (2, 2):
        labels = np.array([
            ['(TP)', '(FN)'],
            ['(FP)', '(TN)']
        ])
        
        # Get label names for display
        reverse_mapping = {v: k for k, v in label_mapping.items()}
        class_names = [reverse_mapping[0], reverse_mapping[1]]
        
        # Create annotation array properly
        annot = np.empty_like(labels, dtype=object)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                annot[i, j] = f"{labels[i, j]}\n{cm[i, j]}"

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=annot, fmt='', cmap="Blues",
                    xticklabels=[f"Predicted {class_names[0]}", f"Predicted {class_names[1]}"],
                    yticklabels=[f"Actual {class_names[0]}", f"Actual {class_names[1]}"])
    else:
        # Handle multi-class case
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix for Premise vs. Conclusion Classification')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")

def evaluate_model(model_path, data_path, results_path):
    """Main evaluation function"""
    
    # Create results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    
    # Load test data and create label mapping
    print("Loading test data...")
    texts, labels, label_mapping = load_test_data(data_path)
    print(f"Total samples: {len(texts)}")
    
    # Create dataset and dataloader
    test_dataset = TextDataset(texts, labels, tokenizer, label_mapping)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Make predictions
    print("Making predictions...")
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(true_labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_true_labels, all_predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Create target names for classification report
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    target_names = [reverse_mapping[i] for i in sorted(reverse_mapping.keys())]
    
    # Generate classification report
    print("Generating classification report...")
    class_report = classification_report(
        all_true_labels, 
        all_predictions, 
        target_names=target_names,
        digits=4
    )
    
    # Save classification report
    report_path = os.path.join(results_path, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Test Data Path: {data_path}\n")
        f.write(f"Total Samples: {len(texts)}\n")
        f.write(f"Label Mapping: {label_mapping}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(class_report)
    
    print(f"Classification report saved to: {report_path}")
    print("\nClassification Report:")
    print(class_report)
    
    # Plot and save confusion matrix
    cm_path = os.path.join(results_path, "confusion_matrix.png")
    plot_confusion_matrix(all_true_labels, all_predictions, cm_path, label_mapping)
    
    # Print confusion matrix values
    cm = confusion_matrix(all_true_labels, all_predictions)
    print(f"\nConfusion Matrix:")
    if cm.shape == (2, 2):
        print(f"True Positives (TP): {cm[0][0]}")
        print(f"False Negatives (FN): {cm[0][1]}")
        print(f"False Positives (FP): {cm[1][0]}")
        print(f"True Negatives (TN): {cm[1][1]}")
    else:
        print("Confusion Matrix:")
        print(cm)
    
    return accuracy, class_report, cm

if __name__ == "__main__":
    # Run evaluation
    print("Starting model evaluation...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Data path: {DATA_PATH}")
    print(f"Results will be saved to: {RESULTS_PATH}")
    
    try:
        accuracy, report, cm = evaluate_model(MODEL_PATH, DATA_PATH, RESULTS_PATH)
        print("\nEvaluation completed successfully!")
        print(f"Final Accuracy: {accuracy:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
