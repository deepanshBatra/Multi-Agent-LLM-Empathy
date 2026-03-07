import pandas as pd

def load_data_from_csv(file_path):
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.
    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

# if __name__ == "__main__":
#     # Example usage
#     file_path = "data/test_sent_emo.csv"
#     data = load_data(file_path)
#     if data is not None:
#         print(data.head())