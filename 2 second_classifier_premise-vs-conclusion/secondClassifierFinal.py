import os
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.calibration import calibration_curve

class FocalLossTrainer(Trainer):
    def __init__(self, alpha=None, gamma=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
    
        ce_loss = F.cross_entropy(logits, labels.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        alpha = self.alpha.to(labels.device)[labels.long()]
        focal_loss = (alpha * (1 - pt) ** self.gamma * ce_loss).mean()
    
        return (focal_loss, outputs) if return_outputs else focal_loss

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = F.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)
    
    return {
        "f1": f1_score(labels, preds, average="binary"),
        "precision": precision_score(labels, preds, average="binary"),
        "recall": recall_score(labels, preds, average="binary"),
        "roc_auc": roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds).tolist()
    }

class TrainingLogger(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

            with open(self.log_path, "a") as f:
                f.write(f"Step {state.global_step}:\n")
                for k, v in logs.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")

class MetricLogger(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
    def on_evaluate(self, args, state, control, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Loss curves
        train_loss = [log['loss'] for log in state.log_history if 'loss' in log]
        eval_loss = [log['eval_loss'] for log in state.log_history if 'eval_loss' in log]
        
        plt.figure()
        plt.plot(train_loss, label='Training')
        plt.plot(np.linspace(0, len(train_loss)-1, len(eval_loss)), eval_loss, label='Validation')
        plt.title('Training Dynamics')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'loss_curves.png'))
        plt.close()

def save_model_artifacts(trainer, output_dir, test_data, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model and tokenizer
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Generate test predictions
    preds = trainer.predict(test_data)
    probs = F.softmax(torch.tensor(preds.predictions), dim=1).numpy()
    
    # Confusion Matrix with labels
    cm = confusion_matrix(preds.label_ids, np.argmax(preds.predictions, axis=1))
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Premises', 'Predicted Conclusions'],
                yticklabels=['Actual Premises', 'Actual Conclusions'])
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # ROC Curve
    RocCurveDisplay.from_predictions(preds.label_ids, probs[:, 1])
    plt.title('ROC Curve')
    plt.savefig(os.path.join(output_dir, 'roc_auc.png'))
    plt.close()
    
    # Calibration Curve with labels
    prob_true, prob_pred = calibration_curve(preds.label_ids, probs[:, 1], n_bins=10)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'calibration.png'))
    plt.close()

# Training configuration
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
    remove_unused_columns=False,
    save_total_limit=1
)

MODELS = {
    "BERT": "bert-base-uncased",
    "LegalBERT": "nlpaueb/legal-bert-base-uncased",
    "InCaseLawBERT": "law-ai/InCaseLawBERT",
    "RoBERTa": "roberta-base"
}

for model_name, checkpoint in MODELS.items():
    output_dir = f"{model_name}_prem_conc_finetuned"
    log_file = os.path.join(output_dir, "training_log.txt")
    
    # Load fresh dataset for each model
    raw_dataset = load_dataset("csv", data_files="csv_files/*.csv", split="train")
    raw_dataset = raw_dataset.map(
        lambda x: {"labels": 0 if x["label"] == "prem" else 1},
        remove_columns=["label"]
    )
    
    # Dataset splitting
    split = raw_dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = split["test"].train_test_split(test_size=0.5, seed=42)
    final_dataset = DatasetDict({
        "train": split["train"],
        "valid": test_valid["train"],
        "test": test_valid["test"]
    })
    
    # Class weights calculation
    train_labels = final_dataset["train"]["labels"]
    class_counts = np.bincount(train_labels)
    total = len(train_labels)
    alpha = torch.tensor([total/class_counts[0], total/class_counts[1]], dtype=torch.float32)
    
    # Model initialization
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
    
    # Tokenization
    tokenized_datasets = final_dataset.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=512),
        batched=True
    ).remove_columns(["text"])
    
    # Initialize callbacks
    callbacks = [
        MetricLogger(output_dir),
        TrainingLogger(log_file)
    ]
    
    # Training
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics,
        alpha=alpha,
        gamma=2,
        callbacks=callbacks
    )
    
    trainer.train()
    save_model_artifacts(trainer, output_dir, tokenized_datasets["test"], tokenizer)
