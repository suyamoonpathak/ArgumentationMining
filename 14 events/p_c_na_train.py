import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed,
)
from torch.nn import functional as F
import matplotlib.pyplot as plt
import json
# --- MODIFICATION START: Added ast for safe string-to-list conversion ---
import ast
# --- MODIFICATION END ---

# Get absolute paths based on script location
ALL_DATA_DIR = Path("./updated_events/p_c_na_with_events_updated")
RESULTS_DIR = Path("./DATA5/3 prem_vs_conc_na results/results_p_c_na_with_events_updated")

# Focal Loss Configuration
FOCAL_PARAMS = {"gamma": 2.0, "alpha": None}
REGULARIZATION = {"dropout_rate": 0.3, "weight_decay": 0.1}

class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        labels = labels.long()
        
        ce_loss = F.cross_entropy(
            logits, 
            labels, 
            reduction='none',
            weight=FOCAL_PARAMS["alpha"].to(labels.device) if FOCAL_PARAMS["alpha"] is not None else None
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** FOCAL_PARAMS["gamma"] * ce_loss).mean()
        
        return (focal_loss, outputs) if return_outputs else focal_loss

class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.train_loss = []
        self.val_loss = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                self.train_loss.append(logs["loss"])
            if "eval_loss" in logs:
                self.val_loss.append(logs["eval_loss"])

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

# --- MODIFICATION START: New function to preprocess the 'events' column ---
def preprocess_events(example):
    """
    Parses the string representation of a list in the 'events' column
    and joins the items into a single string.
    """
    try:
        # ast.literal_eval safely parses the string into a Python list
        events_list = ast.literal_eval(example["events"])
        # Ensure the parsed object is a list before joining
        if isinstance(events_list, list):
            example["processed_text"] = " ; ".join(events_list)
        else:
            example["processed_text"] = ""
    except (ValueError, SyntaxError):
        # Handle cases where the string is malformed or empty
        example["processed_text"] = ""
    return example
# --- MODIFICATION END ---

# --- MODIFICATION START: Made create_dirs robust to TypeError ---
def create_dirs(model_name, fold):
    """Creates the necessary directories for a given model and fold."""
    # Ensure RESULTS_DIR is a Path object before use
    results_path = Path(RESULTS_DIR)
    
    base_dir = results_path / model_name / f"fold_{fold}"
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_dir / "plots").mkdir(parents=True, exist_ok=True)
    return base_dir
# --- MODIFICATION END ---

# Metrics and visualization functions
def save_metrics(predictions, labels, output_dir):
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "precision_weighted": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
        "recall_weighted": recall_score(labels, predictions, average="weighted", zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0)
    }

    with open(f"{output_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def save_classification_report(predictions, labels, output_dir):
    report = classification_report(
        labels, predictions,
        target_names=["Non-Argumentative", "Premise", "Conclusion"],
        zero_division=0
    )

    with open(f"{output_dir}/classification_report.txt", "w") as f:
        f.write(report)

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=["Non-Argumentative", "Premise", "Conclusion"],
                yticklabels=["Non-Argumentative", "Premise", "Conclusion"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Three-Way Classification')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_focal_loss_curves(train_loss, val_loss, output_path):
    min_length = min(len(train_loss), len(val_loss))
    train_loss_clipped = train_loss[:min_length]
    val_loss_clipped = val_loss[:min_length]
    epochs = list(range(min_length))
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss_clipped, label='Training Loss', marker='o', linestyle='-', color='b')
    plt.plot(epochs, val_loss_clipped, label='Validation Loss', marker='x', linestyle='--', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Focal Loss")
    plt.title("Training/Validation Focal Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

# Main function
def main():
    print(ALL_DATA_DIR)
    all_files = list(ALL_DATA_DIR.glob("*.csv"))
    assert len(all_files) == 40, f"Expected 40 files, found {len(all_files)}"

    train_files, test_files = train_test_split(
        all_files, test_size=8, random_state=SEED
    )

    # Load datasets
    train_dataset = load_dataset("csv", data_files=[str(f) for f in train_files])["train"]
    test_dataset = load_dataset("csv", data_files=[str(f) for f in test_files])["train"]

    # --- MODIFICATION START: Preprocess 'events' and rename 'class' column ---
    print("Preprocessing 'events' column to create input text...")
    # Use the new function to process the 'events' column into 'processed_text'
    train_dataset = train_dataset.map(preprocess_events, num_proc=4)
    test_dataset = test_dataset.map(preprocess_events, num_proc=4)

    # Rename the 'class' column to 'labels' which is expected by Hugging Face
    train_dataset = train_dataset.rename_column("class", "labels")
    test_dataset = test_dataset.rename_column("class", "labels")
    # --- MODIFICATION END ---

    # Cross-validation loop
    for model_name, model_checkpoint in MODELS.items():
        print(f"\n{'='*40}")
        print(f"Training {model_name}")
        print(f"{'='*40}\n")

        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        # --- MODIFICATION START: Update tokenize function to use 'processed_text' ---
        def tokenize_function(examples):
            # Tokenize the newly created 'processed_text' column
            return tokenizer(
                examples["processed_text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        # --- MODIFICATION END ---
        
        # --- MODIFICATION START: Update map call to remove now-unused columns ---
        # Process training data by tokenizing and removing old text columns
        tokenized_train = train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text", "events", "processed_text"]
        )
        # --- MODIFICATION END ---

        class_counts = np.bincount(tokenized_train["labels"])
        FOCAL_PARAMS["alpha"] = torch.tensor([1/(c/len(tokenized_train)) for c in class_counts], dtype=torch.float32)
        print(f"Class distribution: {class_counts}")
        print(f"Focal loss alpha weights: {FOCAL_PARAMS['alpha']}")

        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        
        best_eval_loss = float('inf')
        best_model_path = None
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(tokenized_train, tokenized_train["labels"])):
            fold_num = fold + 1
            print(f"\nStarting Fold {fold_num} for {model_name}")

            model = AutoModelForSequenceClassification.from_pretrained(
                model_checkpoint,
                num_labels=3,
                hidden_dropout_prob=REGULARIZATION["dropout_rate"],
                attention_probs_dropout_prob=REGULARIZATION["dropout_rate"]
            )

            fold_dir = create_dirs(model_name, fold_num)

            training_args = TrainingArguments(
                output_dir=str(fold_dir / "checkpoints"),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=1e-5,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                num_train_epochs=EPOCHS,
                weight_decay=REGULARIZATION["weight_decay"],
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                seed=SEED,
                fp16=False,
                report_to="none",
                max_grad_norm=0.3,
                label_smoothing_factor=0.2,
                save_total_limit=1,
            )

            loss_history = LossHistoryCallback()
            trainer = FocalLossTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train.select(train_idx),
                eval_dataset=tokenized_train.select(val_idx),
                compute_metrics=lambda p: {
                    "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
                    "f1_macro": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="macro"),
                    "f1_weighted": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="weighted")
                },
                callbacks=[loss_history]
            )

            trainer.train()
            plot_focal_loss_curves(
                loss_history.train_loss,
                loss_history.val_loss,
                str(fold_dir / "plots" / "focal_loss_curves.png")
            )

            val_pred = trainer.predict(tokenized_train.select(val_idx))
            val_preds = np.argmax(val_pred.predictions, axis=1)
            final_eval_loss = val_pred.metrics['test_loss']

            save_metrics(val_preds, val_pred.label_ids, str(fold_dir))
            save_classification_report(val_preds, val_pred.label_ids, str(fold_dir))
            
            # --- MODIFICATION START: Corrected call to plot_confusion_matrix ---
            plot_confusion_matrix(val_pred.label_ids, val_preds, str(fold_dir / "plots" / "confusion_matrix.png"))
            # --- MODIFICATION END ---

            if final_eval_loss < best_eval_loss:
                best_eval_loss = final_eval_loss
                best_model_dir = RESULTS_DIR / model_name / "best_model"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                
                trainer.save_model(str(best_model_dir))
                tokenizer.save_pretrained(str(best_model_dir))
                
                best_model_path = str(best_model_dir)
                print(f"New best model saved! Fold {fold_num}, Eval Loss: {final_eval_loss:.4f}")

        print(f"\n{model_name} Cross-Validation Complete!")
        if best_model_path:
            print(f"Best model saved at: {best_model_path}")

            print(f"\nEvaluating best {model_name} model on test set...")
            best_model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
            
            # --- MODIFICATION START: Tokenize test set with updated functions ---
            tokenized_test = test_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text", "events", "processed_text"]
            )
            # --- MODIFICATION END ---

            test_trainer = FocalLossTrainer(
                model=best_model,
                args=TrainingArguments(output_dir="./temp_test", per_device_eval_batch_size=BATCH_SIZE),
                tokenizer=tokenizer
            )

            test_pred = test_trainer.predict(tokenized_test)
            test_preds = np.argmax(test_pred.predictions, axis=1)

            test_dir = RESULTS_DIR / model_name / "test_evaluation"
            test_dir.mkdir(parents=True, exist_ok=True)
            save_metrics(test_preds, test_pred.label_ids, str(test_dir))
            save_classification_report(test_preds, test_pred.label_ids, str(test_dir))
            
            # --- MODIFICATION START: Corrected call for the final test set as well ---
            plot_confusion_matrix(test_pred.label_ids, test_preds, str(test_dir / "confusion_matrix.png"))
            # --- MODIFICATION END ---

if __name__ == "__main__":
    main()