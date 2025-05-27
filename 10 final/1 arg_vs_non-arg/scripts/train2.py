import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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
import seaborn as sns
import itertools
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
    EarlyStoppingCallback
)
import matplotlib.pyplot as plt
import json

from pathlib import Path

# Get absolute paths based on script location
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
ALL_DATA_DIR = DATA_DIR / "all"
RESULTS_DIR = SCRIPT_DIR.parent / "DATA5/1 arg_vs_non-arg results/results_fourth_run"

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
NUM_FOLDS = 10
BATCH_SIZE = 4
EPOCHS = 3

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

# Metrics and visualization functions (keeping your existing functions)
def save_metrics(predictions, labels, output_dir):
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, average="binary"),
        "recall": recall_score(labels, predictions, average="binary"),
        "f1": f1_score(labels, predictions, average="binary")
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def save_classification_report(predictions, labels, output_dir):
    report = classification_report(
        labels, predictions,
        target_names=["Non-Argumentative", "Argumentative"]
    )

    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)

    labels = np.array([
        ['(TP)', '(FN)'],
        ['(FP)', '(TN)']
    ])

    # Fix: Create annotation array properly
    annot = np.empty_like(labels, dtype=object)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            annot[i, j] = f"{labels[i, j]}\n{cm[i, j]}"

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues",
                xticklabels=["Predicted Arg", "Predicted Non-Arg"],
                yticklabels=["Actual Arg", "Actual Non-Arg"])
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix for Argumentative Classification')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_loss_curves(log_history, output_dir):
    train_loss = []
    eval_loss = []

    current_epoch = -1
    epoch_train_losses = []

    for log in log_history:
        if 'loss' in log and 'epoch' in log:
            epoch = int(log['epoch'])
            if epoch > current_epoch:
                if epoch_train_losses: 
                    train_loss.append(np.mean(epoch_train_losses))
                current_epoch = epoch
                epoch_train_losses = []
            epoch_train_losses.append(log['loss'])

        elif 'eval_loss' in log:
            eval_loss.append(log['eval_loss'])

    if epoch_train_losses:
        train_loss.append(np.mean(epoch_train_losses))

    min_epochs = min(len(train_loss), len(eval_loss))
    train_loss = train_loss[:min_epochs]
    eval_loss = eval_loss[:min_epochs]

    plt.figure(figsize=(8, 6))
    plt.plot(range(min_epochs), train_loss, label='Training Loss', marker='o', linestyle='-', color='b')
    plt.plot(range(min_epochs), eval_loss, label='Validation Loss', marker='x', linestyle='--', color='orange')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/plots/loss_curves.png")
    plt.close()

# Main function
def main():
    # Load all files from the 'all' directory
    print(ALL_DATA_DIR)
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

        # Track best model across all folds
        best_eval_loss = float('inf')
        best_model_path = None
        best_fold = None
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(
            tokenized_train, tokenized_train["labels"]
        )):
            fold_num = fold + 1
            print(f"\nStarting Fold {fold_num} for {model_name}")

            # Create fresh model instance for each fold
            model = AutoModelForSequenceClassification.from_pretrained(
                model_checkpoint,
                num_labels=2,
                hidden_dropout_prob=0.2,
                attention_probs_dropout_prob=0.2
            )

            # Create directories
            fold_dir = create_dirs(model_name, fold_num)

            # Configure trainer with fold-specific settings
            training_args = TrainingArguments(
                output_dir=str(fold_dir / "checkpoints"),
                evaluation_strategy="epoch",
                save_strategy="epoch",  # Save at each epoch
                learning_rate=1e-5,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_train_epochs=EPOCHS,
                weight_decay=0.3,
                load_best_model_at_end=True,  # Load best model at end
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                seed=SEED,
                fp16=False,
                report_to="none",
                max_grad_norm=0.3,
                label_smoothing_factor=0.2,
                save_total_limit=1,  # Keep only best 2 checkpoints
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
            plot_loss_curves(trainer.state.log_history, fold_dir)

            # Generate validation reports
            val_pred = trainer.predict(tokenized_train.select(val_idx))
            val_preds = np.argmax(val_pred.predictions, axis=1)

            # Use the validation loss from the prediction results instead
            final_eval_loss = val_pred.metrics['test_loss']

            # Save fold metrics
            fold_metrics = save_metrics(val_preds, val_pred.label_ids, str(fold_dir))
            save_classification_report(val_preds, val_pred.label_ids, str(fold_dir))
            plot_confusion_matrix(val_preds, val_pred.label_ids, str(fold_dir / "confusion_matrix.png"))

            # Check if this is the best model so far
            if final_eval_loss < best_eval_loss:
                best_eval_loss = final_eval_loss
                best_fold = fold_num
                
                # Save the best model
                best_model_dir = RESULTS_DIR / model_name / "best_model"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                
                # Save model and tokenizer
                trainer.save_model(str(best_model_dir))
                tokenizer.save_pretrained(str(best_model_dir))
                
                best_model_path = str(best_model_dir)
                print(f"New best model saved! Fold {fold_num}, Eval Loss: {final_eval_loss:.4f}")

            print(f"\n{model_name} Cross-Validation Complete!")
            if best_model_path:
                print(f"Best model saved at: {best_model_path}")

            # Final evaluation on held-out test set using best model
            if best_model_path:
                print(f"\nEvaluating best {model_name} model on test set...")
                
                # Load the best model for testing
                best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)

        tokenized_test = test_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )

        test_trainer = Trainer(
            model=best_model,
            args=TrainingArguments(output_dir="./temp"),
            tokenizer=tokenizer
        )

        test_pred = test_trainer.predict(tokenized_test)
        test_preds = np.argmax(test_pred.predictions, axis=1)

        test_dir = RESULTS_DIR / model_name / "test_evaluation"
        test_dir.mkdir(parents=True, exist_ok=True)
        save_metrics(test_preds, test_pred.label_ids, str(test_dir))
        save_classification_report(test_preds, test_pred.label_ids, str(test_dir))
        plot_confusion_matrix(test_preds, test_pred.label_ids, str(test_dir / "confusion_matrix.png"))

if __name__ == "__main__":
    main()
