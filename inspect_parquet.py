# ml_service/inspect_parquet.py
import pandas as pd
import config 

print(f"Attempting to load Parquet file from: {config.PROCESSED_DATA_PATH}")

try:
    df_check = pd.read_parquet(config.PROCESSED_DATA_PATH)
    
    print("\nSuccessfully loaded Parquet file!")
    print(f"Shape of the DataFrame: {df_check.shape}")
    
    print(f"\nFull list of columns ({len(df_check.columns)} total):")
    all_columns = df_check.columns.tolist()
    # Print all columns to make it easy to search
    print(all_columns) 

    # Example: Check for one-hot encoded 'home_ownership' columns
    home_ownership_dummies = [col for col in all_columns if col.startswith('home_ownership_')]
    if home_ownership_dummies:
        print("\n>>> Found one-hot encoded 'home_ownership' columns:")
        print(home_ownership_dummies)
    else:
        print("\n>>> NOTE: Did not find one-hot encoded 'home_ownership' columns starting with 'home_ownership_'.")
        if 'home_ownership' in all_columns:
             print("    However, an original 'home_ownership' column exists. Check its dtype and values.")
             print("    Dtype:", df_check['home_ownership'].dtype)
             print("    Value Counts:", df_check['home_ownership'].value_counts(dropna=False).head())


except FileNotFoundError:
    print(f"\nERROR: The Parquet file was not found at {config.PROCESSED_DATA_PATH}")
    print("Please ensure 'build_dataset.py' ran successfully and created this file.")
except Exception as e:
    print(f"\nAn error occurred while reading or inspecting the Parquet file: {e}")