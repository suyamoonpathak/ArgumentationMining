import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from datasets import load_dataset, Dataset, ClassLabel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback, set_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import json
from torch.nn import functional as F
import seaborn as sns

# Configure paths
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR.parent / "results"

class Config:
    SEED = 42
    NUM_FOLDS = 10
    BATCH_SIZE = 4
    EPOCHS = 3
    MODELS = {
        # "BERT": "bert-base-uncased",
        # "InCaseLawBERT": "law-ai/InCaseLawBERT",
        "LegalBERT": "nlpaueb/legal-bert-base-uncased",
        "RoBERTa": "roberta-base"
    }
    MAX_SEQ_LENGTH = 512
    FOCAL_PARAMS = {"gamma": 2.0, "alpha": None}
    REGULARIZATION = {"dropout_rate": 0.3, "weight_decay": 0.1}

set_seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)

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
        dataset = load_dataset("csv", data_files=str(doc_path))["train"]
        if not self.label_map:
            # Create consistent label mapping across all documents
            self.label_map = ClassLabel(names=sorted(set(dataset["label"])))
        int_labels = [self.label_map.str2int(l) for l in dataset["label"]]
        return max(set(int_labels), key=int_labels.count)

class FocalLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Convert labels to long tensor
        labels = labels.long()
        
        # Calculate Focal Loss
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
            # Capture training loss
            if "loss" in logs:
                self.train_loss.append(logs["loss"])
            # Capture validation loss
            if "eval_loss" in logs:
                self.val_loss.append(logs["eval_loss"])

def plot_loss_curves(train_loss, val_loss, output_path):
    # Find the minimum length between the two lists
    min_length = min(len(train_loss), len(val_loss))
    
    # Clip both lists to the same length
    train_loss_clipped = train_loss[:min_length]
    val_loss_clipped = val_loss[:min_length]
    
    # Create x-axis values (epochs)
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
    cv = DocumentLevelCV(DATA_DIR / "all")
    train_files, test_files = cv.train_test_split()
    
    # Load and encode datasets
    def load_encoded(files):
        ds = load_dataset("csv", data_files=[str(f) for f in files])["train"]
        return ds.map(lambda x: {"labels": cv.label_map.str2int(x["label"])}, remove_columns=["label"])
    
    train_ds = load_encoded(train_files)
    test_ds = load_encoded(test_files)

    for model_name, checkpoint in Config.MODELS.items():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        
        # Tokenize datasets
        def tokenize(batch):
            return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=Config.MAX_SEQ_LENGTH)
        
        tokenized_train = train_ds.map(tokenize, batched=True, remove_columns=["text"])
        tokenized_test = test_ds.map(tokenize, batched=True, remove_columns=["text"])

        # Calculate class weights
        class_counts = np.bincount(tokenized_train["labels"])
        Config.FOCAL_PARAMS["alpha"] = torch.tensor([1/(c/len(tokenized_train)) for c in class_counts], dtype=torch.float32)

        # Cross-validation
        skf = StratifiedKFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)
        for fold, (train_idx, val_idx) in enumerate(skf.split(tokenized_train, tokenized_train["labels"])):
            # Convert indices to Python integers
            train_idx = [int(i) for i in train_idx]
            val_idx = [int(i) for i in val_idx]

            # Initialize model
            model = AutoModelForSequenceClassification.from_pretrained(
                checkpoint, num_labels=2,
                hidden_dropout_prob=Config.REGULARIZATION["dropout_rate"],
                attention_probs_dropout_prob=Config.REGULARIZATION["dropout_rate"]
            )

            # Training setup
            training_args = TrainingArguments(
                output_dir=RESULTS_DIR / model_name / f"fold_{fold+1}",
                evaluation_strategy="epoch",
                save_strategy="no",
                per_device_train_batch_size=Config.BATCH_SIZE,
                per_device_eval_batch_size=Config.BATCH_SIZE,
                num_train_epochs=Config.EPOCHS,
                weight_decay=Config.REGULARIZATION["weight_decay"],
                metric_for_best_model="f1",
                greater_is_better=True,
                seed=Config.SEED
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
                callbacks=[loss_history]
            )
            
            trainer.train()
            
            final_train_metrics = trainer.evaluate(tokenized_train.select(train_idx))
            if "eval_loss" in final_train_metrics:
                loss_history.train_loss.append(final_train_metrics["eval_loss"])
            
            plot_loss_curves(
                loss_history.train_loss,
                loss_history.val_loss,
                RESULTS_DIR / model_name / f"fold_{fold+1}" / "loss_curves.png"
            )


            
            # Save results
            preds = trainer.predict(tokenized_test)
            probs = F.softmax(torch.tensor(preds.predictions), dim=1).numpy()
            save_results(preds.label_ids, np.argmax(preds.predictions, axis=1), probs, RESULTS_DIR / model_name / f"fold_{fold+1}")

def plot_confusion_matrix(y_true, y_pred, output_path):
    cm = confusion_matrix(y_true, y_pred)
    cell_labels = np.array([
        [f"(TP)\n{cm[0,0]}", f"(FN)\n{cm[0,1]}"],
        [f"(FP)\n{cm[1,0]}", f"(TN)\n{cm[1,1]}"]
    ])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=cell_labels, fmt='', cmap="Blues",
                xticklabels=["Predicted Conclusion", "Predicted Premise"],
                yticklabels=["Actual Conclusion", "Actual Premise"])
    plt.xlabel('Prediction')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix for Argumentative Classification')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()




def save_results(true, pred, probs, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": accuracy_score(true, pred),
        "precision": precision_score(true, pred, average="binary"),
        "recall": recall_score(true, pred, average="binary"),
        "f1": f1_score(true, pred, average="binary")
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)
    
    # Generate and save classification report to a text file
    report = classification_report(
        true, 
        pred, 
        target_names=["Conclusion", "Premise"],
        digits=4
    )
    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)
    
    plot_confusion_matrix(true, pred, out_dir / "confusion_matrix.png")


if __name__ == "__main__":
    main()
