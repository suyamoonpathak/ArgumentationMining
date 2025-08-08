import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, set_seed, EarlyStoppingCallback
import matplotlib.pyplot as plt
import json
from torch.nn import functional as F
import seaborn as sns
# --- MODIFICATION START: Added ast for safe string-to-list conversion ---
import ast
# --- MODIFICATION END ---


# Configure paths
ALL_DATA_DIR = Path("./updated_events/prem_vs_conc_with_events_updated")
RESULTS_DIR = Path("./DATA5/2 prem_vs_conc results/results_prem_vs_conc_with_events_updated")

class Config:
    SEED = 42
    NUM_FOLDS = 10
    BATCH_SIZE = 4
    EPOCHS = 10
    MODELS = {
        "BERT": "bert-base-uncased",
        "InCaseLawBERT": "law-ai/InCaseLawBERT",
        "LegalBERT": "nlpaueb/legal-bert-base-uncased",
        "RoBERTa": "roberta-base"
    }
    MAX_SEQ_LENGTH = 512
    FOCAL_PARAMS = {"gamma": 2.0, "alpha": None}
    REGULARIZATION = {"dropout_rate": 0.3, "weight_decay": 0.1}

set_seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)

# --- MODIFICATION START: New function to preprocess the 'events' column ---
def preprocess_events(example):
    """
    Parses the string representation of a list in the 'events' column
    and joins the items into a single string.
    """
    try:
        # ast.literal_eval safely parses the string into a Python list
        events_list = ast.literal_eval(example["events"])
        if isinstance(events_list, list):
            example["processed_text"] = " ; ".join(events_list)
        else:
            example["processed_text"] = ""
    except (ValueError, SyntaxError):
        # Handle cases where the string is malformed or empty
        example["processed_text"] = ""
    return example
# --- MODIFICATION END ---

class DocumentLevelCV:
    def __init__(self, data_dir: Path):
        self.documents = sorted(data_dir.glob("*.csv"))
        self.label_map = None
        
    def train_test_split(self):
        """Stratified 80/20 split using document majority classes"""
        doc_labels = [self._get_doc_label(doc) for doc in self.documents]
        return train_test_split(self.documents, test_size=8, stratify=doc_labels, random_state=Config.SEED)
    
    def _get_doc_label(self, doc_path: Path):
        """Get integer label for document using majority class"""
        # --- FIX: Removed hyphen before 'quiet' ---
        dataset = load_dataset("csv", data_files=str(doc_path))["train"]
        if not self.label_map:
            self.label_map = ClassLabel(names=sorted(set(dataset["label"])))
        int_labels = [self.label_map.str2int(l) for l in dataset["label"]]
        return max(set(int_labels), key=int_labels.count)

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
            weight=Config.FOCAL_PARAMS["alpha"].to(labels.device)
        )
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** Config.FOCAL_PARAMS["gamma"] * ce_loss).mean()
        
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

def plot_loss_curves(train_loss, val_loss, output_path):
    min_length = min(len(train_loss), len(val_loss))
    train_loss_clipped = train_loss[:min_length]
    val_loss_clipped = val_loss[:min_length]
    epochs = list(range(min_length))
    
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss_clipped, label='Training Loss', marker='o', linestyle='-', color='b')
    plt.plot(epochs, val_loss_clipped, label='Validation Loss', marker='x', linestyle='--', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def main():
    cv = DocumentLevelCV(ALL_DATA_DIR)
    train_files, test_files = cv.train_test_split()
    
    def load_encoded(files):
        # --- FIX: Removed hyphen before 'quiet' ---
        ds = load_dataset("csv", data_files=[str(f) for f in files])["train"]
        # --- FIX: Removed hyphen before 'num_proc' ---
        return ds.map(lambda x: {"labels": cv.label_map.str2int(x["label"])}, remove_columns=["label"], num_proc=4)
    
    train_ds = load_encoded(train_files)
    test_ds = load_encoded(test_files)
    
    print("Preprocessing 'events' column to create input text...")
    # --- FIX: Removed hyphens before 'num_proc' ---
    train_ds = train_ds.map(preprocess_events, num_proc=4)
    test_ds = test_ds.map(preprocess_events, num_proc=4)

    # ... the rest of the main function remains the same ...
    for model_name, checkpoint in Config.MODELS.items():
        print(f"\n{'='*40}")
        print(f"Training {model_name}")
        print(f"{'='*40}\n")
        
        best_f1_score = 0.0
        best_fold = None
        best_model_path = None
        
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        def tokenize(batch):
            return tokenizer(batch["processed_text"], padding="max_length", truncation=True, max_length=Config.MAX_SEQ_LENGTH)
        
        tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["text", "events", "processed_text"])
        tokenized_test = test_ds.map(tokenize, batched=True, remove_columns=["text", "events", "processed_text"])

        class_counts = np.bincount(tokenized_train["labels"])
        Config.FOCAL_PARAMS["alpha"] = torch.tensor([1/(c/len(tokenized_train)) for c in class_counts], dtype=torch.float32)

        skf = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
        for fold, (train_idx, val_idx) in enumerate(skf.split(tokenized_train, tokenized_train["labels"])):
            fold_num = fold + 1
            print(f"\nStarting Fold {fold_num} for {model_name}")
            
            train_idx, val_idx = [int(i) for i in train_idx], [int(i) for i in val_idx]

            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, num_labels=2,
                hidden_dropout_prob=Config.REGULARIZATION["dropout_rate"],
                attention_probs_dropout_prob=Config.REGULARIZATION["dropout_rate"]
            )

            fold_dir = RESULTS_DIR / model_name / f"fold_{fold_num}"
            training_args = TrainingArguments(
                output_dir=str(fold_dir),
                evaluation_strategy="epoch",
                save_strategy="epoch",
                per_device_train_batch_size=Config.BATCH_SIZE,
                per_device_eval_batch_size=Config.BATCH_SIZE,
                num_train_epochs=Config.EPOCHS,
                load_best_model_at_end=True,
                weight_decay=Config.REGULARIZATION["weight_decay"],
                metric_for_best_model="eval_loss",  
                greater_is_better=False,
                seed=Config.SEED,
                save_total_limit=1,
                report_to="none"
            )
            
            loss_history = LossHistoryCallback()
            trainer = FocalLossTrainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train.select(train_idx),
                eval_dataset=tokenized_train.select(val_idx),
                compute_metrics=lambda p: {
                    "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1), average="binary"),
                    "roc_auc": roc_auc_score(p.label_ids, F.softmax(torch.tensor(p.predictions), dim=1)[:, 1].numpy())
                },
                callbacks=[loss_history, EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            trainer.train()
            
            val_pred = trainer.predict(tokenized_train.select(val_idx))
            val_preds = np.argmax(val_pred.predictions, axis=1)
            current_f1 = f1_score(val_pred.label_ids, val_preds, average="binary")
            
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                best_fold = fold_num
                
                best_model_dir = RESULTS_DIR / model_name / "best_model"
                trainer.save_model(str(best_model_dir))
                tokenizer.save_pretrained(str(best_model_dir))
                best_model_path = best_model_dir
                
                with open(best_model_dir / "best_fold_info.json", "w") as f:
                    json.dump({
                        "best_fold": best_fold,
                        "best_f1_score": float(best_f1_score)
                    }, f, indent=2)
            
            plot_loss_curves(loss_history.train_loss, loss_history.val_loss, fold_dir / "loss_curves.png")
            
            probs = F.softmax(torch.tensor(val_pred.predictions), dim=1).numpy()
            save_results(val_pred.label_ids, val_preds, probs, fold_dir)
            
        print(f"\nBest {model_name} model: Fold {best_fold} with F1 score: {best_f1_score:.4f}")

        if best_model_path:
            print(f"\nEvaluating {model_name} BEST model on test set...")
            best_model = AutoModelForSequenceClassification.from_pretrained(str(best_model_path))
            test_trainer = FocalLossTrainer(
                model=best_model,
                args=TrainingArguments(output_dir="./temp", report_to="none"),
                tokenizer=tokenizer
            )

            test_preds = test_trainer.predict(tokenized_test)
            test_probs = F.softmax(torch.tensor(test_preds.predictions), dim=1).numpy()
            test_dir = RESULTS_DIR / model_name / "test_evaluation"
            save_results(test_preds.label_ids, np.argmax(test_preds.predictions, axis=1), test_probs, test_dir)
# --- MODIFICATION START: Corrected TP/TN labels in confusion matrix ---
def plot_confusion_matrix(y_true, y_pred, output_path):
    # Assumes 0: Conclusion, 1: Premise
    cm = confusion_matrix(y_true, y_pred)
    
    # Correctly map confusion matrix cells to TP, TN, FP, FN
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0,0], cm[0,1], cm[1,0], cm[1,1])
    
    cell_labels = np.array([
        [f"(TN)\n{tn}", f"(FP)\n{fp}"],
        [f"(FN)\n{fn}", f"(TP)\n{tp}"]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=cell_labels, fmt='', cmap="Blues",
                xticklabels=["Predicted Conclusion", "Predicted Premise"],
                yticklabels=["Actual Conclusion", "Actual Premise"])
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix for Premise vs. Conclusion')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
# --- MODIFICATION END ---

def save_results(true, pred, probs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": accuracy_score(true, pred),
        "precision": precision_score(true, pred, average="binary", zero_division=0),
        "recall": recall_score(true, pred, average="binary", zero_division=0),
        "f1": f1_score(true, pred, average="binary", zero_division=0)
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    
    report = classification_report(
        true, 
        pred, 
        target_names=["Conclusion", "Premise"],
        digits=4,
        zero_division=0
    )
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    plot_confusion_matrix(true, pred, out_dir / "confusion_matrix.png")

if __name__ == "__main__":
    main()