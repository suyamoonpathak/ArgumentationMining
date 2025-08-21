import csv
import os
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from datetime import datetime
import time

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure model and tokenizer
cache_dir = "/DATA5/suyamoon/argmining/huggingface_cache"  
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # Replace with your model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
model.to(device)

# System prompt to instruct LLaMA
system_prompt = """You are a Harvard-trained legal expert specialized in legal argument analysis. Your job is to decide the relationship between two legal arguments: the SOURCE and the TARGET.

## TASK: Read the SOURCE and TARGET arguments below and decide if the SOURCE supports the TARGET, attacks the TARGET, or has no relation to it. This is important. Your answer will help judges in serious cases. A wrong answer could cause mistakes in how arguments are understood.

## DEFINITIONS:
- SUPPORT: SOURCE gives reasons, evidence, or legal rules that make TARGET stronger or more believable.
- ATTACK: SOURCE gives reasons, evidence, or rules that make TARGET weaker or less believable.
- NO-RELATION: SOURCE and TARGET are about different issues or facts, and the SOURCE does not affect TARGET at all.

## SIGNS OF SUPPORT (examples only):
- SOURCE agrees with TARGET’s point.
- SOURCE uses reasoning or facts that help TARGET.
- SOURCE cites law or precedent that matches TARGET’s position.

## SIGNS OF ATTACK (examples only):
- SOURCE disagrees with TARGET.
- SOURCE gives facts or law that go against TARGET.
- SOURCE shows why TARGET is wrong.

## SIGNS OF NO-RELATION:
- SOURCE talks about something totally different.
- SOURCE would not change TARGET’s strength if removed.
- No clear link between the two.

### LIMITATION:
- Do NOT just look for the sample words. Think about the meaning and the relationship.
- Only respond with one lowercase word: "support", "attack", "no-relation"
"""

def generate_prediction(source_text, target_text):
    prompt = f""" 
    ## SOURCE ARGUMENT: "{source_text}"
    ## TARGET ARGUMENT: "{target_text}"
    Your Response:
    """
    
    # Prepare the input for the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Adjust for your model
    ]

    try:
        # Generate output from the model
        outputs = model.generate(
            input_ids.input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        response = outputs[0][input_ids.input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True).strip().lower()
    except Exception as e:
        print(f"⚠️ Error: {str(e)}")
        return "no-relation"  # Default fallback

def process_csv_files():
    # Create output directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)
    
    # Get all CSV files from the test folder
    csv_files = glob.glob('test/*.csv')
    
    if not csv_files:
        print("No CSV files found in the 'test' folder!")
        return
    
    print(f"Found {len(csv_files)} CSV files to process")

    for csv_file in csv_files:
        print(f"\n🔄 Processing file: {csv_file}")
        
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        if 'source_text' not in df.columns or 'target_text' not in df.columns or 'relation' not in df.columns:
            print(f"❌ Skipping {csv_file}: Missing 'source_text', 'target_text', or 'relation' columns")
            continue
        
   