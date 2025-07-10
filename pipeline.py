import pandas as pd

# Step 1: Data Ingestion
def ingest_data(file_path):
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error ingesting data: {e}")

# Step 2: Data Cleaning
def clean_data(data):
    try:
        data.drop_duplicates(inplace=True)
        data.fillna(data.mean(), inplace=True)
        return data
    except Exception as e:
        print(f"Error cleaning data: {e}")

# Step 3: Data Transformation
def transform_data(data):
    try:
        # Perform transformations here (e.g., feature scaling, encoding)
        return data
    except Exception as e:
        print(f"Error transforming data: {e}")

# Step 4: Data Storage
def store_data(data, output_file_path):
    try:
        data.to_csv(output_file_path, index=False)
    except Exception as e:
        print(f"Error storing data: {e}")

# Main pipeline function
def run_pipeline(input_file_path, output_file_path):
    data = ingest_data(input_file_path)
    data = clean_data(data)
    data = transform_data(data)
    store_data(data, output_file_path)

# Run the pipeline
input_file_path = "input.csv"
output_file_path = "output.csv"
run_pipeline(input_file_path, output_file_path)
