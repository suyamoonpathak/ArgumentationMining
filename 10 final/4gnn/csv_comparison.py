import pandas as pd
import os

def combine_fold_results():
    """Combine multiple fold comparison CSV files into one"""
    
    # Define the CSV files
    file_names = [
        "fold_comparison_results_processed_early_stopped.csv",
        "fold_comparison_results_processed.csv", 
        "fold_comparison_results_raw_early_stopped.csv",
        "fold_comparison_results_raw.csv"
    ]
    
    combined_data = []
    
    for file in file_names:
        # Check if file exists
        if not os.path.exists(file):
            print(f"Warning: {file} not found, skipping...")
            continue
            
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            print(f"Processing {file}...")
            
            # Extract model name from file name
            # Remove 'fold_comparison_results_' prefix and '.csv' suffix
            model_name = file.replace('fold_comparison_results_', '').replace('.csv', '')
            
            # Select required columns and create a copy
            temp_df = df[['fold_number', 'accuracy', 'f1_score_macro_avg']].copy()
            
            # Add model_name from file name
            temp_df['model_name'] = model_name
            
            # Rename f1_score_macro_avg to f1_score
            temp_df.rename(columns={'f1_score_macro_avg': 'f1_score'}, inplace=True)
            
            # Reorder columns to match desired format
            temp_df = temp_df[['model_name', 'fold_number', 'accuracy', 'f1_score']]
            
            combined_data.append(temp_df)
            print(f"Added {len(temp_df)} rows from {file}")
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not combined_data:
        print("No data to combine!")
        return None
    
    # Concatenate all dataframes
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Sort by model_name and fold_number for better organization
    combined_df = combined_df.sort_values(['model_name', 'fold_number']).reset_index(drop=True)
    
    # Save to new CSV file
    output_file = 'combined_fold_comparison_results.csv'
    combined_df.to_csv(output_file, index=False, float_format='%.4f')
    
    print(f"\nCombined results saved to: {output_file}")
    print(f"Total rows: {len(combined_df)}")
    print(f"Models included: {combined_df['model_name'].unique().tolist()}")
    
    # Display summary
    print("\nPreview of combined data:")
    print(combined_df.head(10))
    
    # Show summary statistics by model
    print("\nSummary by model:")
    summary = combined_df.groupby('model_name').agg({
        'accuracy': ['mean', 'std', 'min', 'max'],
        'f1_score': ['mean', 'std', 'min', 'max']
    }).round(4)
    print(summary)
    
    return combined_df

if __name__ == "__main__":
    combined_df = combine_fold_results()
