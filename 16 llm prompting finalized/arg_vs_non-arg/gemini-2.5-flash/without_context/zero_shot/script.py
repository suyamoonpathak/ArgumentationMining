import csv
import google.generativeai as genai
import time
import os
import glob
from datetime import datetime, timedelta
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

api_keys = ["KEY1", "KEY2", "KEY3"]  # Replace with your actual API keys

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

def get_argumentative_prediction(text):
    global current_key_index
    select_next_available_key()
    
    model = genai.GenerativeModel('gemini-2.5-flash')
    current_key = api_keys[current_key_index]
    
    try:
        prompt = f"""You are a Harvard-trained legal expert specialized in legal argumentation analysis, with deep expertise in distinguishing "argumentative" from "non-argumentative" text in legal case files.

            ## TASK: Read the sentence and decide if it is argumentative or non-argumentative. Your analysis is very important. Your answer will be used in a legal research tool for senior judges working on high-impact cases. A wrong answer could cause legal mistakes, unfair judgments, or even wrongful convictions. So be very careful and precise.

            ## DEFINITIONS:
            - ARGUMENTATIVE: A text that gives reasoning, makes conclusions, applies law to facts, supports or challenges a position, or criticizes another argument. It often uses cause-and-effect words, expresses opinions, gives judgments, or refers to legal precedent as part of reasoning.
            - NON-ARGUMENTATIVE: A sentence that only states facts, describes procedures, quotes laws or prior rulings without analysis, or gives administrative/clerical details.

            ## ARGUMENTATIVE CLUES TO LOOK FOR (Examples Only):
            1. Words that show an opinion, conclusion, or reasoning. Examples: therefore, thus, hence, argue, contend, assert, believe, maintain, show, prove, opinion, judgment, position, argument, claim, conclusion.
            2. Words that judge something as good or bad. Examples: correct, valid, justified, fair, legal, unfair, wrong, incorrect, invalid, flawed, unfounded, inconsistent.

            ## NON-ARGUMENTATIVE CLUES TO LOOK FOR (Examples Only):
            1. Just facts, dates, or events without reasoning.
            2. Direct quotes of law or testimony without any analysis.
            3. Descriptions of court procedures (e.g., filing a motion).

            ### LIMITATION:
            - Do NOT rely only on the above words. They are just examples. Look at whether the sentence’s main purpose is to make an argument or just state information.
            - Only give your final answer as one word in lowercase:"argumentative" or "non-argumentative"

            TEXT TO ANALYZE:
            "{text}"

            Your Response:
            """

        
        # Update key usage
        now = datetime.now()
        key_usage[current_key]['count'] += 1
        key_usage[current_key]['last_used'] = now
        key_usage[current_key]['request_times'].append(now)
        
        response = model.generate_content(prompt)
        prediction = response.text.strip().lower()
        
        # Ensure we get a valid response
        if prediction not in ['argumentative', 'non-argumentative']:
            # Try to extract the classification from the response
            if 'argumentative' in prediction and 'non-argumentative' not in prediction:
                prediction = 'argumentative'
            elif 'non-argumentative' in prediction:
                prediction = 'non-argumentative'
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
        return get_argumentative_prediction(text)

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
        
        if 'text' not in df.columns or 'class' not in df.columns:
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
            actual_class = row['class']
            
            # Convert numerical class to text
            actual_label = 'argumentative' if actual_class == 1 else 'non-argumentative'
            
            log_status(idx + 1, total_rows, os.path.basename(csv_file))
            
            # Get prediction from Gemini
            prediction = get_argumentative_prediction(text)
            
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
