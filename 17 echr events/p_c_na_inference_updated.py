import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
import ast
from tqdm import tqdm

# Set global seeds for reproducibility
SEED = 42
set_seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Configuration
INPUT_DATA_DIR = Path("./updated_echr_events/p_c_na_ucreat_events_echr")
MODEL_PATH = Path("../14 events/DATA5/3 prem_vs_conc_na results/results_p_c_na_with_events_updated/RoBERTa/best_model")
OUTPUT_DIR = Path("./predictions/p_c_na_updated_events_echr/RoBERTa")
BATCH_SIZE = 8

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_events(example):
    """
    Parses the string representation of a list in the 'events' column
    and joins the items into a single string.
    """
    try:
        events_list = ast.literal_eval(example["events"])
        if isinstance(events_list, list):
            example["processed_text"] = " ; ".join(events_list)
        else:
            example["processed_text"] = ""
    except (ValueError, SyntaxError):
        example["processed_text"] = ""
    return example

def load_model_and_tokenizer(model_path):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    
    return model, tokenizer

def predict_on_dataset(model, tokenizer, dataset):
    """Make predictions on the dataset"""
    
    # Tokenization function with proper padding and truncation
    def tokenize_function(examples):
        return tokenizer(
            examples["processed_text"],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors=None
        )
    
    print("Tokenizing dataset...")
    
    # Get original column names to preserve them
    original_columns = dataset.column_names
    
    # Remove any potential label columns and only keep necessary columns for tokenization
    columns_to_remove = [col for col in original_columns if col in ['label', 'labels', 'class']]
    
    # Tokenize the dataset and remove only the processed_text column
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["processed_text"] + columns_to_remove
    )
    
    # Create trainer for prediction
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="./temp_inference",
            per_device_eval_batch_size=BATCH_SIZE,
            dataloader_drop_last=False,
            report_to="none"
        ),
        tokenizer=tokenizer
    )
    
    print("Making predictions...")
    predictions = trainer.predict(tokenized_dataset)
    
    # Get predicted labels (argmax of logits)
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    
    # Get prediction probabilities
    probabilities = torch.nn.functional.softmax(torch.tensor(predictions.predictions), dim=1)
    
    return predicted_labels, probabilities.numpy()

def process_single_file(file_path, model, tokenizer):
    """Process a single CSV file and return predictions"""
    print(f"\nProcessing: {file_path.name}")
    
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(df)
        
        # Apply preprocessing to create processed_text
        dataset = dataset.map(preprocess_events, num_proc=1)
        
        # Check if processed_text was created successfully
        sample_text = dataset[0]["processed_text"]
        print(f"Sample processed text: {sample_text[:100]}...")
        
        # Make predictions
        predicted_labels, probabilities = predict_on_dataset(model, tokenizer, dataset)
        
        # Add predictions to the original dataframe
        df['predictions'] = predicted_labels
        df['prob_non_argumentative'] = probabilities[:, 0]  # Probability for class 0 (Non-Argumentative)
        df['prob_premise'] = probabilities[:, 1]            # Probability for class 1 (Premise)
        df['prob_conclusion'] = probabilities[:, 2]         # Probability for class 2 (Conclusion)
        
        # Map predictions to readable labels (based on training script)
        label_mapping = {0: 'Non-Argumentative', 1: 'Premise', 2: 'Conclusion'}
        df['prediction_label'] = df['predictions'].map(label_mapping)
        
        return df
        
    except Exception as e:
        print(f"Error processing {file_path.name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Check if input directory exists
    if not INPUT_DATA_DIR.exists():
        print(f"Error: Input directory {INPUT_DATA_DIR} does not exist!")
        return
    
    # Check if model path exists
    if not MODEL_PATH.exists():
        print(f"Error: Model path {MODEL_PATH} does not exist!")
        print(f"Expected model path: {MODEL_PATH}")
        return
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
        print("Model and tokenizer loaded successfully!")
        print(f"Model expects {model.config.num_labels} classes")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Get all CSV files in the input directory
    csv_files = list(INPUT_DATA_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    
    if len(csv_files) == 0:
        print(f"No CSV files found in {INPUT_DATA_DIR}")
        return
    
    # Process each file
    successful_predictions = 0
    failed_predictions = 0
    
    for csv_file in tqdm(csv_files, desc="Processing files"):
        # Process the file
        result_df = process_single_file(csv_file, model, tokenizer)
        
        if result_df is not None:
            # Save the results
            output_file = OUTPUT_DIR / csv_file.name
            result_df.to_csv(output_file, index=False)
            
            print(f"Saved predictions to: {output_file}")
            print(f"Prediction distribution:")
            print(result_df['prediction_label'].value_counts())
            
            successful_predictions += 1
        else:
            failed_predictions += 1
    
    # Summary
    print(f"\n{'='*50}")
    print("THREE-CLASS CLASSIFICATION INFERENCE SUMMARY")
    print(f"{'='*50}")
    print(f"Total files: {len(csv_files)}")
    print(f"Successfully processed: {successful_predictions}")
    print(f"Failed: {failed_predictions}")
    print(f"Results saved in: {OUTPUT_DIR}")
    
    if successful_predictions > 0:
        print(f"\nPredictions saved with the following columns:")
        print("- predictions: Numerical predictions (0: Non-Argumentative, 1: Premise, 2: Conclusion)")
        print("- prediction_label: Human-readable prediction labels")
        print("- prob_non_argumentative: Probability for Non-Argumentative class")
        print("- prob_premise: Probability for Premise class")
        print("- prob_conclusion: Probability for Conclusion class")
        
        print(f"\nLabel Mapping:")
        print("- 0: Non-Argumentative")
        print("- 1: Premise")
        print("- 2: Conclusion")

if __name__ == "__main__":
    main()
