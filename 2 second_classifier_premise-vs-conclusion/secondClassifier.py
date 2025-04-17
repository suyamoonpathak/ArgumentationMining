import os
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
from sklearn.metrics import f1_score

# Custom Trainer with Focal Loss
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
    
        # Get alpha values for each sample in the batch
        alpha = self.alpha.to(labels.device)[labels.long()]  # Shape: [batch_size]
    
        # Calculate focal loss with per-sample alpha weights
        focal_loss = (alpha * (1 - pt) ** self.gamma * ce_loss).mean()
    
        return (focal_loss, outputs) if return_outputs else focal_loss


# Custom callback for logging
class TrainingLogger(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            with open(self.log_path, "a") as f:
                f.write(f"Step {state.global_step}:\n")
                for k, v in logs.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")

# Load and prepare dataset
dataset = load_dataset("csv", data_files="csv_files/*.csv", split="train")

def encode_labels(example):
    example["labels"] = 0 if example["label"] == "prem" else 1
    return example

dataset = dataset.map(encode_labels)
dataset = dataset.remove_columns(["label"])  # Remove original string column


# Split dataset
split = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = split["test"].train_test_split(test_size=0.5, seed=42)
dataset = DatasetDict({
    "train": split["train"],
    "valid": test_valid["train"], 
    "test": test_valid["test"]
})

# Calculate class weights
train_labels = dataset["train"]["labels"]
class_counts = np.bincount(train_labels)
total = len(train_labels)
alpha = torch.tensor([total/class_counts[0], total/class_counts[1]], dtype=torch.float32)
print("Alpha values:", alpha)


# Model configurations
MODELS = {
    "BERT": "bert-base-uncased",
    "LegalBERT": "nlpaueb/legal-bert-base-uncased",
    "InCaseLawBERT": "law-ai/InCaseLawBERT",
    "RoBERTa": "roberta-base"
}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="no",
    logging_steps=100,
    remove_unused_columns=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, predictions, average="binary")}

for model_name, model_checkpoint in MODELS.items():
    # Create output directory
    output_dir = f"{model_name}_prem_conc_finetuned"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training_log.txt")
    
    # Initialize components
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    
    # Tokenization
    tokenized_datasets = dataset.map(
        lambda ex: tokenizer(ex["text"], padding="max_length", truncation=True, max_length=512),
        batched=True
    ).remove_columns(["text"])
    tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # Train with logging
    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics,
        alpha=alpha,
        gamma=2,
        callbacks=[TrainingLogger(log_file)]
    )
    
    # Execute training
    trainer.train()
    
    # Save final artifacts
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Log test results
    test_results = trainer.evaluate(tokenized_datasets["test"])
    with open(log_file, "a") as f:
        f.write(f"\n\nFinal Test Results:\n{test_results}")