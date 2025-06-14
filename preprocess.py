import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Handle missing values
    data = data.dropna()

    # Encode categorical variables
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Normalize numerical values
    scaler = StandardScaler()
    data[data.select_dtypes(include=['int64', 'float64']).columns] = scaler.fit_transform(data[data.select_dtypes(include=['int64', 'float64']).columns])

    return data, label_encoders, scaler

if __name__ == "__main__":
    file_path = "mushrooms_decoded.csv"
    data, label_encoders, scaler = load_and_preprocess_data(file_path)
    print("Data preprocessing completed.")
