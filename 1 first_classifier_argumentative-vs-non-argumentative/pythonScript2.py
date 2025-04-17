# Import libraries
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import f1_score

# Load dataset from CSV files
dataset = load_dataset(
    "csv", 
    data_dir="combinedCleanFinal",
    delimiter=",",
    split="train"
)

# Split dataset into train, validation, and test sets (80-10-10 split)
split = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = split["test"].train_test_split(test_size=0.5, seed=42)

# Create DatasetDict for Hugging Face compatibility
dataset = DatasetDict({
    "train": split["train"],
    "valid": test_valid["train"], 
    "test": test_valid["test"]
})

# Rename 'class' column to 'labels' for compatibility with Hugging Face models
dataset = dataset.rename_column("class", "labels")

# Load tokenizer for BERT-based models
MODELS = {
    "BERT": "bert-base-uncased",
    "LegalBERT": "nlpaueb/legal-bert-base-uncased",
    "InCaseLawBERT": "law-ai/InCaseLawBERT"
}

tokenizer = AutoTokenizer.from_pretrained(MODELS["BERT"])

# Tokenization function
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=512
    )
    tokenized["labels"] = examples["labels"]  # Preserve labels during tokenization
    return tokenized

# Apply tokenization to the datasets
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns and set format for PyTorch compatibility
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Define metrics function (F1 score)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"f1": f1_score(labels, predictions, average="binary")}

# Training arguments configuration
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
    remove_unused_columns=False  # Ensure label column is not removed automatically
)

# Fine-tune each model
for model_name, model_checkpoint in MODELS.items():
    print(f"Training {model_name}...")
    
    # Load pre-trained model with sequence classification head
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint,
        num_labels=2  # Binary classification task
    )
    
    # Initialize Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
        compute_metrics=compute_metrics
    )
    
    # Train the model and evaluate on the validation set
    trainer.train()
    
    # Evaluate on the test set and print results
    results = trainer.evaluate(tokenized_datasets["test"])
    print(f"{model_name} Results:", results)
    
    # Save the fine-tuned model and tokenizer
    model.save_pretrained(f"./{model_name}_finetuned")
    tokenizer.save_pretrained(f"./{model_name}_finetuned")
