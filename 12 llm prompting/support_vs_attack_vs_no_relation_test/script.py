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

def get_relation_prediction(source_text, target_text):
    global current_key_index
    select_next_available_key()
    
    model = genai.GenerativeModel('gemini-2.0-flash')
    current_key = api_keys[current_key_index]
    
    try:
        prompt = f"""You are a Harvard-trained legal scholar with expertise in legal argumentation analysis. Your task is to analyze the relationship between two legal arguments with the precision and analytical rigor expected in top-tier legal academia.

CLASSIFICATION TASK:
Determine the relationship between the source argument and target argument. The relationship can be "support", "attack", or "no-relation".

DEFINITIONS:
• SUPPORT: The source argument provides evidence, reasoning, or justification that strengthens, reinforces, or validates the target argument. The source helps establish the credibility or validity of the target.
• ATTACK: The source argument contradicts, undermines, refutes, or weakens the target argument. The source challenges the validity or credibility of the target.
• NO-RELATION: The source and target arguments are independent, unrelated, or address different issues without any logical connection that would constitute support or attack.

ANALYTICAL FRAMEWORK:
Look for SUPPORT indicators:
- Source provides evidence for target's claims
- Source establishes legal precedent that validates target
- Source offers reasoning that strengthens target's position
- Logical flow where source builds toward target's conclusion

Look for ATTACK indicators:
- Source contradicts target's claims or reasoning
- Source provides counter-evidence to target
- Source establishes precedent that undermines target
- Logical inconsistency between source and target positions

Look for NO-RELATION indicators:
- Arguments address completely different legal issues
- No logical connection between the reasoning chains
- Independent factual statements without argumentative relationship

EXEMPLARS FOR CALIBRATION:

SUPPORT Examples:
1. Source: "It must be pointed out in this connection that in order to assess a measure under Article 106(2) TFEU, the Commission is not required, contrary to what is claimed by the appellant, to examine whether the conditions laid down by the case-law in Altmark, in particular the second and fourth of those conditions, are met."
   Target: "As the General Court indeed held in paragraph 63 of the judgment under appeal, verification of the conditions laid down in the Altmark case-law occurs upstream, that is to say in the examination of the issue of whether the measures at issue must be characterised as State aid."
   [Analysis: Source establishes that Commission is not required to examine Altmark conditions under Article 106(2) TFEU, which directly supports target's explanation that Altmark verification occurs "upstream" in the State aid characterization phase rather than in the Article 106(2) assessment stage]

2. Source: "Thus, it is clear that, following that line of reasoning, the General Court did not substitute its own grounds for those in the decision at issue but simply interpreted that decision in the light of its actual content."
   Target: "The General Court stated, in paragraph 107 of the judgment under appeal, that 'regrettably, recital 119 of the [decision at issue], which contains the conclusion on the existence of aid within the meaning of Article 107(1) TFEU, was limited to stating that the aid consisted in the reduction of ""the compensation consisting of the employer's contribution"", without mentioning the compensation or over-compensation charges'."
   [Analysis: Source establishes that the General Court properly interpreted the decision based on its actual content rather than substituting new reasoning, which directly supports target's demonstration of this interpretive approach where the Court identified specific limitations in recital 119's language about compensation charges - showing the Court was indeed analyzing what the decision actually stated rather than creating new grounds]

ATTACK Examples:
1. Source: "Consequently, the General Court should have examined the possibility of de facto selectivity."
   Target: "Accordingly, the second part of the third ground of appeal is unfounded."
   [Analysis: Source asserts that the General Court failed to conduct a necessary examination of de facto selectivity, directly contradicting target's conclusion that the appeal ground is unfounded - if additional examination was required, the appeal cannot be dismissed as lacking merit]

2. Source: "Secondly, it argues that the General Court's reasoning leads to the selectivity of a measure being assessed differently, depending on whether the national legislature decided to create a separate tax or to modify a general tax, and, therefore, depending on the regulatory technique used."
   Target: "It follows that the arguments advanced by the appellant, first, objecting to the present case's being placed on the same footing as that which gave rise to Advocate General Warner's Opinion and, secondly, intended to demonstrate that the objective of the measure at issue was to safeguard the principle of fiscal neutrality, and not to solve a specific problem, are insufficient to invalidate the General Court's reasoning and are, therefore, ineffective."
   [Analysis: Source presents substantive criticism that the General Court's approach creates inconsistent selectivity assessments based on legislative technique, directly undermining target's dismissal of appellant's arguments as "insufficient" and "ineffective" - the source demonstrates potential flaws in the Court's methodology that target seeks to reject]

NO-RELATION Examples:
1. Source: "In that regard, even in cases where it is apparent from the circumstances under which it was granted that the aid is liable to affect trade between Member States and to distort or threaten to distort competition, the Commission must at least set out those circumstances in the statement of reasons for its decision (see Portugal v Commission, paragraph 89 and the case-law cited)."
   Target: "Following an economic crisis in Korea and Taiwan, the planned projects were not implemented in those countries."
   [Analysis: Source addresses Commission procedural obligations to provide reasoning when State aid affects EU trade and competition, while target describes factual consequences of economic crises in Asian countries on project implementation - these involve entirely different legal frameworks (EU State aid procedural requirements vs. international economic developments) with no argumentative relationship between Commission disclosure duties and Asian project cancellations]
   
2. Source: "at issue must confer a selective advantage on the recipient, that are called into question."
Target: "minimum mining fee payable for natural gas fields put into production before 1  January 1998."
[Analysis: Source discusses the legal requirement for measures to confer selective advantage on recipients in State aid analysis, while target refers to specific mining fee regulations for natural gas fields with a temporal cutoff date - these address completely different legal domains (State aid law vs. natural resource taxation) with no logical connection between selective advantage concepts and mining fee structures]

SOURCE ARGUMENT:
"{source_text}"

TARGET ARGUMENT:
"{target_text}"

INSTRUCTIONS:
Apply your legal training to analyze how the source argument relates to the target argument. Consider whether the source strengthens, weakens, or has no bearing on the target's position.

OUTPUT FORMAT:
Respond with exactly one word: "support" or "attack" or "no-relation"

Your classification:"""

        # Update key usage
        now = datetime.now()
        key_usage[current_key]['count'] += 1
        key_usage[current_key]['last_used'] = now
        key_usage[current_key]['request_times'].append(now)
        
        response = model.generate_content(prompt)
        prediction = response.text.strip().lower()
        
        # Ensure we get a valid response
        if prediction not in ['support', 'attack', 'no-relation']:
            # Try to extract the classification from the response
            if 'support' in prediction:
                prediction = 'support'
            elif 'attack' in prediction:
                prediction = 'attack'
            elif 'no-relation' in prediction or 'no relation' in prediction:
                prediction = 'no-relation'
            else:
                prediction = 'no-relation'  # Default fallback
        
        return prediction
    
    except Exception as e:
        print(f"⚠️ API Error: {str(e)}")
        # Rollback failed request
        if key_usage[current_key]['request_times']:
            key_usage[current_key]['count'] -= 1
            key_usage[current_key]['request_times'].pop()
        current_key_index = (current_key_index + 1) % len(api_keys)
        genai.configure(api_key=api_keys[current_key_index])
        return get_relation_prediction(source_text, target_text)

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
                writer.writerow(['source_text', 'target_text', 'actual_relation', 'predicted_relation'])

        
        total_rows = len(df)
        
        # Process rows starting from where we left off
        for idx in range(start_idx, total_rows):
            row = df.iloc[idx]
            source_text = row['source_text']
            target_text = row['target_text']
            actual_relation = row['relation']

            log_status(idx + 1, total_rows, os.path.basename(csv_file))

            # Get prediction from Gemini
            prediction = get_relation_prediction(source_text, target_text)

            print(f"Source: {source_text[:50]}...")
            print(f"Target: {target_text[:50]}...")
            print(f"Actual: {actual_relation} | Predicted: {prediction}")

            # Immediately append this row to the CSV file
            with open(output_filename, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([source_text, target_text, actual_relation, prediction])

            
            print(f"💾 Row {idx + 1} saved to {output_filename}")
            
            # Small delay to be respectful to the API
            time.sleep(0.5)
        
        print(f"✅ Completed processing: {output_filename}")

if __name__ == "__main__":
    print("🚀 Starting Legal Text Classification with Gemini")

    process_csv_files()
