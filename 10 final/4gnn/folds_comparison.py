import os
import re
import pandas as pd

# Define the root directory (change this to your actual root folder path)
root_dir = './GNN4_10_epochs'  # Update this as needed

# Prepare a list to store results
data = []

# Regex patterns to extract metrics from classification_report.txt
macro_avg_pattern = re.compile(r'macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)')
accuracy_pattern = re.compile(r'accuracy\s+([\d.]+)')

for model_name in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_name)
    if os.path.isdir(model_path):
        for fold in range(1, 11):
            fold_name = f'fold_{fold}'
            fold_path = os.path.join(model_path, fold_name)
            report_path = os.path.join(fold_path, 'classification_report.txt')
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    content = f.read()
                    # Extract macro avg metrics
                    macro_match = macro_avg_pattern.search(content)
                    accuracy_match = accuracy_pattern.search(content)
                    if macro_match and accuracy_match:
                        precision = float(macro_match.group(1))
                        recall = float(macro_match.group(2))
                        f1_score = float(macro_match.group(3))
                        accuracy = float(accuracy_match.group(1))
                        data.append({
                            'model_name': model_name,
                            'fold_number': fold,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1-score': f1_score
                        })

# Create DataFrame and save to CSV
results_df = pd.DataFrame(data)
results_df.to_csv('classification_report_summary_6_10_epochs.csv', index=False)
