import pandas as pd
import random
import os
import json

try:
    print("Reading data from local parquet file...")
    
    # Use the correct filename
    parquet_file = "hackaprompt.parquet"
    
    # Check if file exists
    if not os.path.exists(parquet_file):
        raise FileNotFoundError(f"Parquet file '{parquet_file}' not found in the current directory")
    
    # Load the parquet file
    df_raw = pd.read_parquet(parquet_file)
    print(f"Successfully loaded parquet file with {len(df_raw)} rows")
    
    # Display the columns to help debugging
    print(f"Columns in the dataset: {df_raw.columns.tolist()}")
    
    # Filter for successful attacks
    # Adjust the column name if different in your parquet file
    if 'success' in df_raw.columns:
        malicious_df = df_raw[df_raw['success'] == True]
        print(f"Found {len(malicious_df)} successful attacks")
    else:
        print("Warning: 'success' column not found. Using all available prompts.")
        malicious_df = df_raw
    
    # Extract prompts and ensure we have the right column
    if 'prompt' in malicious_df.columns:
        prompts = malicious_df['prompt'].tolist()
    else:
        # Try to guess the column name for prompts
        prompt_columns = [col for col in malicious_df.columns if 'prompt' in col.lower()]
        if prompt_columns:
            prompts = malicious_df[prompt_columns[0]].tolist()
            print(f"Using '{prompt_columns[0]}' as the prompt column")
        else:
            # If we can't find a column with 'prompt' in the name, use the first string column
            string_cols = malicious_df.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                prompts = malicious_df[string_cols[0]].tolist()
                print(f"Using '{string_cols[0]}' as the prompt column")
            else:
                raise ValueError("Could not identify a column containing prompts")
    
    # Set a seed for reproducibility
    random.seed(42)
    
    # Sample up to 200 prompts
    sample_size = min(200, len(prompts))
    print(f"Sampling {sample_size} prompts...")
    
    # Shuffle and select prompts
    sampled_prompts = random.sample(prompts, sample_size)
    
    # Create a list of dictionaries (better for JSON)
    output_data = [{"prompt": prompt, "label": "malicious"} for prompt in sampled_prompts]
    
    # Save to JSON
    json_file = "malicious_prompts.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully saved {sample_size} prompts to {json_file}")
    
    # Also create JSONL (JSON Lines) format which is often better for large datasets
    jsonl_file = "malicious_prompts.jsonl"
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Also saved as {jsonl_file} (JSON Lines format)")

except Exception as e:
    print(f"Error occurred: {str(e)}")
    
    # Provide a helpful error message
    print("\nCheck that:")
    print("1. The parquet file exists in the current directory")
    print("2. The file is named 'hackaprompt.parquet'")
    print("3. You have pandas and pyarrow installed: conda install pandas pyarrow")