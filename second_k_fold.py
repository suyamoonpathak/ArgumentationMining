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
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    set_seed
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Absolute path setup
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

# Set global seeds
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

class FocalLossTrainer(Trainer):
    def __init__(self, alpha=None, gamma=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        ce_loss = torch.nn.functional.cross_entropy(
            logits, labels.long(), 
            reduction='none', 
            weight=self.alpha.to(labels.device) if self.alpha else None
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        
        return (focal_loss, outputs) if return_outputs else focal_loss

def create_dirs(model_name, fold):
    base_dir = RESULTS_DIR / model_name / f"fold_{fold}"
    (base_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (base_dir / "plots").mkdir(parents=True, exist_ok=True)
    return base_dir

class TrainingLogger(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            with open(self.log_path, "a") as f:
                f.write(f"Step {state.global_step}:\n")
                for k, v in logs.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")

def save_artifacts(trainer, tokenizer, test_data, output_dir):
    # Predictions
    test_pred = trainer.predict(test_data)
    probs = torch.softmax(torch.tensor(test_pred.predictions), dim=1).numpy()
    preds = np.argmax(test_pred.predictions, axis=1)
    
    # Metrics
    metrics = {
        "accuracy": accuracy_score(test_pred.label_ids, preds),
        "precision": precision_score(test_pred.label_ids, preds, average="binary"),
        "recall": recall_score(test_pred.label_ids, preds, average="binary"),
        "f1": f1_score(test_pred.label_ids, preds, average="binary"),
        "roc_auc": roc_auc_score(test_pred.label_ids, probs[:, 1])
    }
    
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Confusion Matrix
    cm = confusion_matrix(test_pred.label_ids, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Premise', 'Conclusion'],
                yticklabels=['Premise', 'Conclusion'])
    plt.title('Confusion Matrix')
    plt.savefig(output_dir / "confusion_matrix.png")
    plt.close()
    
    # Calibration Curve
    prob_true, prob_pred = calibration_curve(test_pred.label_ids, probs[:, 1], n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.savefig(output_dir / "calibration_curve.png")
    plt.close()

def main():
    # Load data
    train_files = list((DATA_DIR / "train").glob("*.csv"))
    test_files = list((DATA_DIR / "test").glob("*.csv"))
    
    assert len(train_files) == 32 and len(test_files) == 8, "Invalid file split"
    
    raw_train = load_dataset("csv", data_files=[str(f) for f in train_files])["train"]
    raw_test = load_dataset("csv", data_files=[str(f) for f in test_files])["train"]
    
    # Preprocess
    raw_train = raw_train.rename_column("label", "labels")
    raw_test = raw_test.rename_column("label", "labels")

    for model_name, checkpoint in MODELS.items():
        print(f"\n{'='*40}")
        print(f"Training {model_name}")
        print(f"{'='*40}\n")
        
        # Tokenization
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        def tokenize_fn(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_train = raw_train.map(
            tokenize_fn, 
            batched=True,
            remove_columns=["text"]
        )
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(
            tokenized_train, tokenized_train["labels"]
        ), 1):
            fold_dir = create_dirs(model_name, fold)
            print(f"\nStarting Fold {fold} for {model_name}")
            
            # Class weights
            fold_labels = [tokenized_train[i]["labels"] for i in train_idx]
            class_counts = np.bincount(fold_labels)
            alpha = torch.tensor([
                len(fold_labels)/class_counts[0], 
                len(fold_labels)/class_counts[1]
            ], dtype=torch.float32) if min(class_counts) > 0 else None
            
            # Model
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, 
                num_labels=2
            )
            
            # Training args
            training_args = TrainingArguments(
                output_dir=str(fold_dir),
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
                logging_dir=str(fold_dir / "logs"),
                logging_steps=50,
                report_to="none",
                seed=SEED,
                fp16=True
            )
            
            # Trainer
            trainer = FocalLossTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train.select(train_idx),
                eval_dataset=tokenized_train.select(val_idx),
                compute_metrics=lambda p: {
                    "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
                    "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="binary")
                },
                alpha=alpha,
                gamma=2,
                callbacks=[TrainingLogger(str(fold_dir / "logs.txt"))]
            )
            
            # Train
            trainer.train()
            trainer.save_model(str(fold_dir / "checkpoints"))
            
            # Fold evaluation
            val_pred = trainer.predict(tokenized_train.select(val_idx))
            val_metrics = {
                "accuracy": accuracy_score(val_pred.label_ids, np.argmax(val_pred.predictions, axis=1)),
                "f1": f1_score(val_pred.label_ids, np.argmax(val_pred.predictions, axis=1), average="binary")
            }
            
            with open(fold_dir / "val_metrics.json", "w") as f:
                json.dump(val_metrics, f, indent=2)
        
        # Final test evaluation
        print(f"\nFinal Test Evaluation for {model_name}")
        tokenized_test = raw_test.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"]
        )
        
        test_trainer = FocalLossTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(RESULTS_DIR / "temp"),
                per_device_eval_batch_size=BATCH_SIZE
            )
        )
        
        test_dir = RESULTS_DIR / model_name / "test_evaluation"
        test_dir.mkdir(parents=True, exist_ok=True)
        save_artifacts(test_trainer, tokenizer, tokenized_test, test_dir)

if __name__ == "__main__":
    main()
