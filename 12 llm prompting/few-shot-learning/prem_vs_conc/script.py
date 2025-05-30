import csv
import google.generativeai as genai
import time
import os
import glob
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Configure API keys
api_keys = ['AIzaSyDUTFotadydDH-jX8xMadQkUU0ZJFoUtrM', 
           'AIzaSyDBMRygr9h2hdTGy2_v5XpVoRKayCoSW3A',
           'AIzaSyA0q1QWVMRXvpW6TyOlb31WwOcSzmQCYyc',
           'AIzaSyCh0WZlL5Onw1YMGPZXVqqbdrkIzR6tRUM',
           'AIzaSyAIjhHTPScZX0j0jYk-iR5VCvRPIj7UwCE',
           'AIzaSyCPWcVheo_YSvKVQCeLvIQbf3viAvGJGXU',
           'AIzaSyC6YiWEgoKZFo9WZT_gwvjtRgR7XA-odd0',
           'AIzaSyBLDaAnFat0Zm_5JFLfkHf-GB1LAXxUYQA'
           ]

key_usage = {key: {'count': 0, 'last_used': None, 'request_times': []} for key in api_keys}
current_key_index = 0
genai.configure(api_key=api_keys[current_key_index])

def log_status(current_text, total, filename):
    current_key = api_keys[current_key_index]
    print(f"\n{'#'*50}")
    print(f"File: {filename} | Processing {current_text}/{total} | Key {current_key_index+1}")
    print(f"Total Requests: {key_usage[current_key]['count']}")
    
    recent_requests = [t for t in key_usage[current_key]['request_times'] 
                      if t > datetime.now() - timedelta(minutes=1)]
    available = 10 - len(recent_requests)
    print(f"Current Key Capacity: {available}/10")
    
    if recent_requests:
        next_available = recent_requests[0] + timedelta(minutes=1)
        wait_time = max(0, (next_available - datetime.now()).total_seconds())
        if available == 0:
            print(f"Cooldown: {wait_time:.1f}s remaining")
    
    print(f"Last Used: {key_usage[current_key]['last_used'].strftime('%H:%M:%S') if key_usage[current_key]['last_used'] else 'Never'}")
    print('#'*50)

calls_per_key = 10

def select_next_available_key():
    global current_key_index
    
    # Try all keys in order
    for i in range(len(api_keys)):
        idx = (current_key_index + i) % len(api_keys)
        key = api_keys[idx]
        
        # Check if this key has been used recently
        recent_requests = [t for t in key_usage[key]['request_times'] 
                          if t > datetime.now() - timedelta(minutes=1)]
        
        # If we haven't made 10 requests in the last minute, use this key
        if len(recent_requests) < calls_per_key:
            current_key_index = idx
            genai.configure(api_key=api_keys[current_key_index])
            return True
    
    # If we get here, all keys are at capacity
    # Find the key that will become available soonest
    earliest_available = None
    earliest_time = datetime.max
    
    for idx, key in enumerate(api_keys):
        if key_usage[key]['request_times']:
            # Get the oldest request within the last minute
            recent_reqs = sorted([t for t in key_usage[key]['request_times'] 
                               if t > datetime.now() - timedelta(minutes=1)])
            if recent_reqs:
                available_time = recent_reqs[0] + timedelta(minutes=1)
                if available_time < earliest_time:
                    earliest_time = available_time
                    earliest_available = idx
    
    if earliest_available is not None:
        wait_time = (earliest_time - datetime.now()).total_seconds()
        if wait_time > 0:
            print(f"\n⏳ All keys at capacity. Waiting {wait_time:.1f} seconds for key {earliest_available+1} to become available...")
            time.sleep(wait_time + 0.5)  # Add a small buffer
        
        current_key_index = earliest_available
        genai.configure(api_key=api_keys[current_key_index])
        return True
    
    return False

def get_premise_conclusion_prediction(text):
    global current_key_index
    select_next_available_key()
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    current_key = api_keys[current_key_index]
    
    try:
        prompt = f"""You are a Harvard-trained legal scholar with expertise in legal argumentation analysis. Your task is to classify argumentative legal text with the precision and analytical rigor expected in top-tier legal academia.

CLASSIFICATION TASK:
Determine whether the given argumentative legal text functions as a "premise" or "conclusion" within legal reasoning.

DEFINITIONS:
• PREMISE: Text that provides supporting evidence, reasoning, legal principles, factual foundations, or intermediate steps that build toward a conclusion. Premises establish the groundwork for legal arguments.
• CONCLUSION: Text that draws final inferences, makes ultimate determinations, renders judgments, or states the logical endpoint of reasoning. Conclusions are what the premises lead to.

ANALYTICAL FRAMEWORK:
Look for these PREMISE indicators:
- Supporting evidence or factual foundations
- Legal principles being established or cited
- Intermediate reasoning steps
- Causal explanations that build arguments
- Background legal context that supports reasoning
- Phrases like "having found," "in view of," "given that"

Look for these CONCLUSION indicators:
- Final determinations or judgments
- Ultimate outcomes of reasoning
- Definitive statements that resolve issues
- Language of finality: "it follows," "consequently," "therefore," "must be rejected"
- Dispositive rulings or decisions

EXEMPLARS FOR CALIBRATION:

PREMISE Examples:
1. "having then found, in paragraph 127 of that judgment, that the payments made in 2008 were independent of those contributions;"
   [Analysis: Establishes factual foundation from prior finding]

2. "In view of the independence of the contributions paid by the farmers in relation to the compensation aid received by them, those contributions cannot be regarded as specific charges imposed on the advantage consisting, in the present case, of the payment of that aid;"
   [Analysis: Provides reasoning based on established facts]

3. "Consequently, that argument must be rejected as inadmissible."
   [Analysis: Intermediate step in reasoning chain]

CONCLUSION Examples:
1. "It follows that the first ground of appeal must be rejected as, in part, inadmissible and, in part, unfounded."
   [Analysis: Final determination using "it follows" with ultimate disposition]

2. "Consequently, the first part of the third ground of appeal must be rejected as unfounded."
   [Analysis: Ultimate ruling on appeal ground]

3. "In the light of the foregoing, the Court considers that there is no need to reopen the oral part of the procedure."
   [Analysis: Final judicial determination based on preceding analysis]

TEXT TO ANALYZE:
"{text}"

INSTRUCTIONS:
Apply your legal training to determine whether this text serves as supporting reasoning (premise) or final determination (conclusion) in the legal argument structure.

OUTPUT FORMAT:
Respond with exactly one word: "premise" or "conclusion"

Your classification:"""

        # Update key usage
        now = datetime.now()
        key_usage[current_key]['count'] += 1
        key_usage[current_key]['last_used'] = now
        key_usage[current_key]['request_times'].append(now)
        
        response = model.generate_content(prompt)
        prediction = response.text.strip().lower()
        
        # Ensure we get a valid response
        if prediction not in ['premise', 'conclusion']:
            # Try to extract the classification from the response
            if 'premise' in prediction and 'conclusion' not in prediction:
                prediction = 'premise'
            elif 'conclusion' in prediction:
                prediction = 'conclusion'
            else:
                prediction = '--'  # Default fallback
        
        return prediction
    
    except Exception as e:
        print(f"⚠️ API Error: {str(e)}")
        # Rollback failed request
        if key_usage[current_key]['request_times']:
            key_usage[current_key]['count'] -= 1
            key_usage[current_key]['request_times'].pop()
        current_key_index = (current_key_index + 1) % len(api_keys)
        genai.configure(api_key=api_keys[current_key_index])
        return get_premise_conclusion_prediction(text)


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
        
        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"❌ Skipping {csv_file}: Missing 'text' or 'class' columns")
            continue
        
        # Create output filename and initialize CSV file with headers
        output_filename = f"predictions/{os.path.basename(csv_file).replace('.csv', '_predictions.csv')}"
        
        # Check if file already exists and has some predictions
        start_idx = 0
        if os.path.exists(output_filename):
            existing_df = pd.read_csv(output_filename)
            start_idx = len(existing_df)
            print(f"📂 Found existing predictions file. Resuming from row {start_idx + 1}")
        else:
            # Create new file with headers
            with open(output_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['text', 'actual_class', 'actual_label', 'predicted_label'])
        
        total_rows = len(df)
        
        # Process rows starting from where we left off
        for idx in range(start_idx, total_rows):
            row = df.iloc[idx]
            text = row['text']
            actual_class = row['label']

            # Convert numerical class to text
            actual_label = 'premise' if actual_class == "premise" else 'conclusion'

            
            log_status(idx + 1, total_rows, os.path.basename(csv_file))
            
            # Get prediction from Gemini
            prediction = get_premise_conclusion_prediction(text)

            
            print(f"Text: {text[:100]}...")
            print(f"Actual: {actual_label} | Predicted: {prediction}")
            
            # Immediately append this row to the CSV file
            with open(output_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([text, actual_class, actual_label, prediction])
            
            print(f"💾 Row {idx + 1} saved to {output_filename}")
            
            # Small delay to be respectful to the API
            time.sleep(0.5)
        
        print(f"✅ Completed processing: {output_filename}")

if __name__ == "__main__":
    print("🚀 Starting Legal Text Classification with Gemini")

    process_csv_files()
