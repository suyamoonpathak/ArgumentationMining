import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

# Configuration
MAX_LEN = 256
BATCH_SIZE = 4
EPOCHS = 3
CLASS_WEIGHTS = {
    'relation': torch.tensor([0.05, 0.3, 0.65]),
    'source': torch.tensor([0.1, 0.2, 0.7]),
    'target': torch.tensor([0.1, 0.2, 0.7])
}

class ArgumentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_len = max_len
        self.label_map = {
            'relation': {'no-relation': 0, 'support': 1, 'attack': 2},
            'source_type': {'na': 0, 'prem': 1, 'conc': 2},
            'target_type':  {'na': 0, 'prem': 1, 'conc': 2},
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row['source_text'] + " </s></s> " + row['target_text']
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'relation_label': torch.tensor(self.label_map['relation'][row['relation']], dtype=torch.long),
            'source_label': torch.tensor(self.label_map['source_type'][row['source_type']], dtype=torch.long),
            'target_label': torch.tensor(self.label_map['target_type'][row['target_type']], dtype=torch.long)
        }

class MultiTaskLegalBERT(torch.nn.Module):
    def __init__(self):
        super(MultiTaskLegalBERT, self).__init__()
        # Load the LegalBERT model using AutoModelForSequenceClassification
        self.legalbert = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.1)
        
        # Create task-specific classifiers
        self.relation_classifier = torch.nn.Linear(768, 3)
        self.source_classifier = torch.nn.Linear(768, 3)
        self.target_classifier = torch.nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        # Get the model's output (this will have 'logits' and 'hidden_states')
        outputs = self.legalbert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Extract the hidden states (last layer hidden states)
        hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)
        
        # We use the first token's hidden state (CLS token)
        pooled_output = hidden_states[:, 0, :]  # Shape: (batch_size, hidden_size)
        
        # Apply dropout for regularization
        pooled_output = self.dropout(pooled_output)
        
        # Return predictions for each task
        return (
            self.relation_classifier(pooled_output),
            self.source_classifier(pooled_output),
            self.target_classifier(pooled_output)
        )


def plot_learning_curves(train_losses, test_losses, output_path):
    epochs = range(1, len(train_losses)+1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-', color='b')
    plt.plot(epochs, test_losses, label='Test Loss', marker='x', linestyle='--', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Test Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, output_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=labels,
                yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_classification_report(y_true_dict, preds_dict, fold_dir):
    with open(os.path.join(fold_dir, 'classification_report.txt'), 'w') as f:
        for task in ['relation', 'source', 'target']:
            labels = ['no-relation', 'support', 'attack'] if task == 'relation' else ['non-arg', 'premise', 'conclusion']
            report = classification_report(y_true_dict[task], preds_dict[task], target_names=labels)
            f.write(f"=== {task.capitalize()} Classification Report ===\n")
            f.write(report)
            f.write("\n\n")
            
df = pd.read_csv('4 PCNA_SANR_final.csv')
sorted_df = df.sort_values('file_name').reset_index(drop=True)
unique_files = sorted_df['file_name'].unique()
kf = KFold(n_splits=10, shuffle=False)

results_dir = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cross-validation loop
for fold_idx, (train_file_idx, test_file_idx) in enumerate(kf.split(unique_files)):
    fold_dir = os.path.join(results_dir, f"fold_{fold_idx+1}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # Data splitting
    train_files = unique_files[train_file_idx]
    test_files = unique_files[test_file_idx]
    train_df = sorted_df[sorted_df['file_name'].isin(train_files)]
    test_df = sorted_df[sorted_df['file_name'].isin(test_files)]
    
    # Model initialization
    model = MultiTaskLegalBERT().to(device)
    optimizer = AdamW([
        {'params': model.legalbert.parameters(), 'lr': 1e-5},
        {'params': model.relation_classifier.parameters(), 'lr': 2e-4},
        {'params': model.source_classifier.parameters(), 'lr': 2e-4},
        {'params': model.target_classifier.parameters(), 'lr': 2e-4}
    ])
    
    relation_criterion = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS['relation'].to(device))
    source_criterion = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS['source'].to(device))
    target_criterion = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS['target'].to(device))

    # Data loaders
    train_loader = DataLoader(ArgumentDataset(train_df, tokenizer, MAX_LEN), 
                             batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(ArgumentDataset(test_df, tokenizer, MAX_LEN), 
                            batch_size=BATCH_SIZE)
    
    # Training variables
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'text'}
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
            
            loss = (0.6 * relation_criterion(outputs[0], inputs['relation_label']) +
                    0.2 * source_criterion(outputs[1], inputs['source_label']) +
                    0.2 * target_criterion(outputs[2], inputs['target_label']))
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss/len(train_loader))
        
        # Testing phase
        model.eval()
        test_loss = 0
        all_preds = {'relation': [], 'source': [], 'target': []}
        all_labels = {'relation': [], 'source': [], 'target': []}
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != 'text'}
                outputs = model(inputs['input_ids'], inputs['attention_mask'])
                
                test_loss += (0.6 * relation_criterion(outputs[0], inputs['relation_label']) +
                              0.2 * source_criterion(outputs[1], inputs['source_label']) +
                              0.2 * target_criterion(outputs[2], inputs['target_label'])).item()
                
                for i, task in enumerate(['relation', 'source', 'target']):
                    all_preds[task].extend(torch.argmax(outputs[i], dim=1).cpu().numpy())
                    all_labels[task].extend(inputs[f'{task}_label'].cpu().numpy())
        
        test_losses.append(test_loss/len(test_loader))
    
    # Save fold results
    plot_learning_curves(train_losses, test_losses, os.path.join(fold_dir, 'learning_curves.png'))
    
    # Confusion matrices
    cm_labels = {
        'relation': ['no-relation', 'support', 'attack'],
        'source': ['non-arg', 'premise', 'conclusion'],
        'target': ['non-arg', 'premise', 'conclusion']
    }
    
    for task in ['relation', 'source', 'target']:
        plot_confusion_matrix(
            all_labels[task], all_preds[task],
            labels=cm_labels[task],
            output_path=os.path.join(fold_dir, f'{task}_confusion_matrix.png'),
            title=f'{task.capitalize()} Confusion Matrix'
        )
    
    save_classification_report(all_labels, all_preds, fold_dir)


print("\nCross-validation completed! Results saved in:", results_dir)
