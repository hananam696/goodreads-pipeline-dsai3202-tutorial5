# goodreads_text_features_trial.py
import pandas as pd
import os
# Set up input and output paths from environment variables (with defaults for local paths in the container)
input_dir = os.environ.get("INPUT_DIR", "/opt/ml/processing/input/features")
output_dir = os.environ.get("OUTPUT_DIR", "/opt/ml/processing/output")
row_chunk = os.environ.get("ROW_CHUNK", "2000") # number of rows to sample (default 2000)
# Load the dataset (assumes a single Parquet file or a directory of Parquet files)
df = pd.read_parquet(input_dir)
# Take the first N rows for the trial
df_subset = df.head(int(row_chunk))
# Save the subset to output in Parquet format
output_path = f"{output_dir}/trial_output.parquet"
df_subset.to_parquet(output_path, index=False)
# Print a confirmation (this will appear in the logs)
print(f"Loaded {len(df)} rows. Saved {len(df_subset)} rows to {output_path}.")