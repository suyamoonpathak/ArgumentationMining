from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import f1_score
import optuna
import os
import csv

# Load and prepare dataset
dataset = load_dataset(
    "csv", 
    data_dir="combinedCleanFinal",
    delimiter=",",
    split="train"
)

split = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = split["test"].train_test_split(test_size=0.5, seed=42)

dataset = DatasetDict({
    "train": split["train"],
    "valid": test_valid["train"], 
    "test": test_valid["test"]
})

dataset = dataset.rename_column("class", "labels")

# Model configurations
MODELS = {
    "BERT": "bert-base-uncased",
    "LegalBERT": "nlpaueb/legal-bert-base-uncased",
    "InCaseLawBERT": "law-ai/InCaseLawBERT"
}

tokenizer = AutoTokenizer.from_pretrained(MODELS["BERT"])

def tokenize_function(examples):
    texts = [str(text) for text in examples["text"]]
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    tokenized["labels"] = examples["labels"]  # Preserve labels during tokenization
    return tokenized


tokenized_datasets = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Add explicit formatting
tokenized_datasets = tokenized_datasets.with_format(
    "torch",
    columns=["input_ids", "attention_mask", "labels"]
)




# Metric computation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, predictions, average="binary")}

def train_and_evaluate(trial, model_checkpoint, model_name):
    # Hyperparameter space
    params = {
        "learning_rate": trial.suggest_categorical("lr", [1e-5, 2e-5, 3e-5]),
        "batch_size": trial.suggest_categorical("batch", [4, 8, 16]),
        "num_epochs": trial.suggest_categorical("epochs", [3, 4, 5]),
        "weight_decay": trial.suggest_categorical("wd", [0.0, 0.01, 0.1]),
        "warmup_steps": trial.suggest_categorical("warmup", [0, 500, 1000]),
    }
    
    # Create trial directory
    trial_dir = f"./hyperparameter_tuning/{model_name}/trial_{trial.number}"
    os.makedirs(trial_dir, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=trial_dir,
        evaluation_strategy="epoch",
        learning_rate=params["learning_rate"],
        per_device_train_batch_size=params["batch_size"],
        per_device_eval_batch_size=8,
        num_train_epochs=params["num_epochs"],
        weight_decay=params["weight_decay"],
        warmup_steps=params["warmup_steps"],
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2
    )
    
    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # Training
    trainer.train()
    
    # Evaluation
    eval_results = trainer.evaluate()
    
    # Save model and tokenizer
    model.save_pretrained(trial_dir)
    tokenizer.save_pretrained(trial_dir)
    
    # Log results
    log_data = {
        "trial": trial.number,
        **params,
        "eval_f1": eval_results["eval_f1"],
        "eval_loss": eval_results["eval_loss"]
    }
    
    log_file = f"./hyperparameter_tuning/{model_name}/trials_log.csv"
    file_exists = os.path.isfile(log_file)
    
    with open(log_file, "a") as f:
        writer = csv.DictWriter(f, fieldnames=log_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_data)
    
    return eval_results["eval_f1"]

# Main execution loop
for model_name, model_checkpoint in MODELS.items():
    print(f"\n{'='*40}\nOptimizing {model_name}\n{'='*40}")
    
    # Create model directory
    os.makedirs(f"./hyperparameter_tuning/{model_name}", exist_ok=True)
    
    # Optimize with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: train_and_evaluate(trial, model_checkpoint, model_name),
        n_trials=15
    )
    
    # Save best trial info
    best_trial = study.best_trial
    print(f"Best trial for {model_name}:")
    print(f" Value: {best_trial.value:.4f}")
    print(" Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best trial config
    with open(f"./hyperparameter_tuning/{model_name}/best_trial.txt", "w") as f:
        f.write(f"Best F1 Score: {best_trial.value:.4f}\n")
        f.write("Hyperparameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"{key}: {value}\n")
