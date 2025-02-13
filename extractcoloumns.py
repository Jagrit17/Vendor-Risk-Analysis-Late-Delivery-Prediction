import pandas as pd

# File paths
data_files = {
    "DataCoSupplyChainDataset": "C:/Programming/Vendor Risk Assessment & Anomaly Detection/DataCoSupplyChainDataset.csv",
    "DescriptionDataCoSupplyChain": "C:/Programming/Vendor Risk Assessment & Anomaly Detection/DescriptionDataCoSupplyChain.csv",
    "TokenizedAccessLogs": "C:/Programming/Vendor Risk Assessment & Anomaly Detection/tokenized_access_logs.csv"
}

# Extract and display columns from each CSV file
for file_name, file_path in data_files.items():
    try:
        df = pd.read_csv(file_path, nrows=5, encoding='ISO-8859-1')  # Use ISO-8859-1 encoding to handle special characters
        print(f"Columns in {file_name}:")
        print(df.columns.tolist())
        print("-" * 50)
    except Exception as e:
        print(f"Error reading {file_name}: {e}")