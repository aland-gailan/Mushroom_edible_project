import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Train the model
def train_and_save_model(data, target_column, model_path):
    # Encode categorical columns
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # Feature selection
    correlation_matrix = data.corr()
    features = correlation_matrix[target_column][correlation_matrix[target_column] > 0.1].index.tolist()

    # Split the data
    X = data[features]
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy}")

    # Save the model
    with open(model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    file_path = "mushrooms_decoded.csv"
    model_path = "mushroom_model.pkl"
    data = pd.read_csv(file_path)
    train_and_save_model(data, "target", model_path)
