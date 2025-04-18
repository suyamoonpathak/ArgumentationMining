import os
import re
import pandas as pd

root_dir = '.'  # Update with your actual root directory path

data = []
report_section_pattern = re.compile(r'=== (.*?) Classification Report ===\n(.*?)(?=\n\n===|\Z)', re.DOTALL)
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
                    
                    # Split into different report sections
                    for match in report_section_pattern.finditer(content):
                        task = match.group(1).strip().split()[0]  # Extract 'Relation', 'Source', or 'Target'
                        report_text = match.group(2)
                        
                        # Extract metrics
                        macro_match = macro_avg_pattern.search(report_text)
                        accuracy_match = accuracy_pattern.search(report_text)
                        
                        if macro_match and accuracy_match:
                            data.append({
                                'model_name': model_name,
                                'fold_number': fold,
                                'task': task,
                                'precision': float(macro_match.group(1)),
                                'recall': float(macro_match.group(2)),
                                'f1-score': float(macro_match.group(3)),
                                'accuracy': float(accuracy_match.group(1))
                            })

# Create and save DataFrame
results_df = pd.DataFrame(data)
results_df = results_df[['model_name', 'fold_number', 'task', 'accuracy', 'precision', 'recall', 'f1-score']]
results_df.to_csv('multi_task_classification_report.csv', index=False)
