import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load the model
def load_model(model_path):
    with open(model_path, "rb") as file:
        return pickle.load(file)

# Streamlit GUI
def main():
    st.title("🍄 Mushroom Classification Prediction")
    st.markdown("""
    ### Welcome to the Mushroom Classification App!
    Enter the features of the mushroom one by one to predict its edibility.
    The app will provide the probability of the mushroom being poisonous.
    """)

    # Load the model
    model_path = "mushroom_model.pkl"
    model = load_model(model_path)

    # Load the dataset for feature names
    file_path = "mushrooms_decoded.csv"
    data = pd.read_csv(file_path)

    # Define the features used during training
    features = ["cap_shape", "cap_surface", "cap_color", "bruises", "odor", "gill_attachment", "gill_spacing", "gill_size", "gill_color"]

    # Encode categorical columns
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

    # User input
    st.markdown("### Mushroom Features")
    # Initialize user_input with None values
    user_input = {feature: None for feature in features}

    # Remove "Select an option" and accompanying space
    # Adjust margins to remove extra space
    st.markdown("<style>.stSelectbox, .stNumberInput {margin-top: -10px;}</style>", unsafe_allow_html=True)

    # Feature selection
    for feature in features:
        st.markdown(f"### **Select {feature.replace('_', ' ').capitalize()}**")
        if feature in label_encoders:
            options = label_encoders[feature].classes_
            user_input[feature] = st.selectbox("", options=options, key=feature)
        else:
            user_input[feature] = st.number_input("", format="%.2f", key=feature)

    st.markdown("### All features selected. Ready to predict!")

    # Predict only if all features are selected
    st.markdown("### Prediction")
    if st.button("Predict 🍄"):
        if None in user_input.values():
            st.error("Please select values for all features before predicting.")
        else:
            input_data = []
            for feature in features:
                if feature in label_encoders:
                    input_data.append(label_encoders[feature].transform([user_input[feature]])[0])
                else:
                    input_data.append(user_input[feature])
            input_data = np.array([input_data]).reshape(1, -1)
            probability = model.predict_proba(input_data)[0][0]  # Probability of being poisonous
            probability_percentage = probability * 100
            classification = "Poisonous" if probability > 0.5 else "Edible"
            st.markdown(f"**Prediction:** {classification}")
            st.markdown(f"**Probability of it being poisonous:** {probability_percentage:.2f}%")

if __name__ == "__main__":
    main()
