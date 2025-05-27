import os
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed
)
import matplotlib.pyplot as plt
import json

# Get absolute paths based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
ALL_DATA_DIR = DATA_DIR / "all"  # New directory containing all 40 files
RESULTS_DIR = SCRIPT_DIR.parent / "DATA5/1 arg_vs_non-arg results/results_10_folds_second_run"

# Set global seeds for reproducibility
SEED = 42
set_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Configuration
MODELS = {
    "BERT": "bert-base-uncased",
    "LegalBERT": "nlpaueb/legal-bert-base-uncased",
    "InCaseLawBERT": "law-ai/InCaseLawBERT",
    "RoBERTa": "roberta-base"
}
NUM_FOLDS = 5
BATCH_SIZE = 4
EPOCHS = 5  

# Create directory structure
def create_dirs(model_name, fold):
    base_dir = RESULTS_DIR / model_name / f"fold_{fold}"
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_dir / "plots").mkdir(parents=True, exist_ok=True)
    return base_dir

# Custom callback for logging
class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            with open(self.log_path / "logs.txt", "a") as f:
                f.write(f"Step {state.global_step}\n")
                f.write(f"Logs: {logs}\n\n")

# Metrics and visualization functions
def save_metrics(predictions, labels, output_dir):
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="binary"),
        "recall": recall_score(labels, predictions, average="binary"),
        "f1": f1_score(labels, predictions, average="binary")
    }
    
    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

def save_classification_report(predictions, labels, output_dir):
    report = classification_report(
        labels, predictions, 
        target_names=["Non-Argumentative", "Argumentative"]
    )
    
    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)

def plot_confusion_matrix(predictions, labels, output_dir):
    cm = confusion_matrix(labels, predictions)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xlabel='Predicted label',
        ylabel='True label',
        xticks=[0, 1],
        yticks=[0, 1],
        title='Confusion Matrix'
    )
    
    plt.savefig(f"{output_dir}/plots/confusion_matrix.png")
    plt.close()

def plot_loss_curves(log_history, output_dir):
    train_loss = [x['loss'] for x in log_history if 'loss' in x]
    eval_loss = [x['eval_loss'] for x in log_history if 'eval_loss' in x]
    
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(eval_loss, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f"{output_dir}/plots/loss_curves.png")
    plt.close()

# Training arguments
def get_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        report_to="none",
        seed=SEED,
        dataloader_num_workers=4,
        fp16=True,
        save_total_limit=1
    )

# Main function
def main():
    # Load all files from the 'all' directory
    all_files = list(ALL_DATA_DIR.glob("*.csv"))
    assert len(all_files) == 40, f"Expected 40 files, found {len(all_files)}"
    
    # Split into document-level train/test (32/8)
    train_files, test_files = train_test_split(
        all_files, 
        test_size=8, 
        random_state=SEED
    )
    
    # Load datasets
    train_dataset = load_dataset("csv", data_files=[str(f) for f in train_files])["train"]
    test_dataset = load_dataset("csv", data_files=[str(f) for f in test_files])["train"]
    
    # Prepare test set
    test_dataset = test_dataset.rename_column("class", "labels")
    
    # Cross-validation loop
    for model_name, model_checkpoint in MODELS.items():
        print(f"\n{'='*40}")
        print(f"Training {model_name}")
        print(f"{'='*40}\n")
        
        # Tokenize once per model
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        # Process training data
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        ).rename_column("class", "labels")
        
        # Document-level stratified K-Fold
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(
            tokenized_train, tokenized_train["labels"]
        )):
            fold_num = fold + 1
            print(f"\nStarting Fold {fold_num} for {model_name}")
            
            # Create fresh model instance for each fold
            model = AutoModelForSequenceClassification.from_pretrained(
                model_checkpoint,
                num_labels=2
            )
            
            # Create directories
            fold_dir = create_dirs(model_name, fold_num)
            
            # Configure trainer with fold-specific settings
            training_args = TrainingArguments(
                output_dir=str(fold_dir),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=2e-5,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_train_epochs=EPOCHS,  # 5 epochs per fold
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model="eval_f1",
                greater_is_better=True,
                seed=SEED,
                fp16=True,
                report_to="none"
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train.select(train_idx),
                eval_dataset=tokenized_train.select(val_idx),
                compute_metrics=lambda p: {
                    "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
                    "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="binary")
                }
            )
            
            # Train and save
            trainer.train()
            trainer.save_model(str(fold_dir / "checkpoints"))
            
            # Generate validation reports
            val_pred = trainer.predict(tokenized_train.select(val_idx))
            val_preds = np.argmax(val_pred.predictions, axis=1)
            
            save_metrics(val_preds, val_pred.label_ids, str(fold_dir))
            save_classification_report(val_preds, val_pred.label_ids, str(fold_dir))
            plot_confusion_matrix(val_preds, val_pred.label_ids, str(fold_dir))
        
        # Final evaluation on held-out test set
        print(f"\nEvaluating {model_name} on test set...")
        tokenized_test = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        test_trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir="./temp"),
            tokenizer=tokenizer
        )
        
        test_pred = test_trainer.predict(tokenized_test)
        test_preds = np.argmax(test_pred.predictions, axis=1)
        
        test_dir = RESULTS_DIR / model_name / "test_evaluation"
        test_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(test_preds, test_pred.label_ids, str(test_dir))
        save_classification_report(test_preds, test_pred.label_ids, str(test_dir))
        plot_confusion_matrix(test_preds, test_pred.label_ids, str(test_dir))


if __name__ == "__main__":
    main()
