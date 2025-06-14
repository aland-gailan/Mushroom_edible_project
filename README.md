# Mushroom Classification Project

This project implements an end-to-end machine learning solution for mushroom classification based on various features. The system uses a trained model to predict whether a mushroom is edible or poisonous.

## Project Overview

- **Go through the Report PDF file first!**
- **Data Preprocessing**: Handle missing values, encode categorical variables, and normalize numerical values.
- **Model Training**: Train a Random Forest Classifier and evaluate its performance.
- **Interactive GUI**: Build a Streamlit-based GUI for real-time predictions.

## Dataset Information

- **Source**: Mushroom dataset (mushrooms_decoded.csv)
- **Description**: Contains features of mushrooms and their edibility.
- **Features**: Various categorical and numerical attributes.
- **Target**: Binary classification (Edible/Poisonous).

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/mushroom-classification-project.git
   cd mushroom-classification-project
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Notebook**
   - Open `mushroom_project.ipynb` in Jupyter Notebook or VS Code.

5. **Launch the GUI**
   ```bash
   streamlit run gui_app.py
   ```

## Features

### 1. Data Preprocessing
- Handles missing values.
- Encodes categorical variables.
- Normalizes numerical values.

### 2. Model Training
- Trains a Random Forest Classifier.
- Evaluates model performance.

### 3. GUI Application
- **Responsive design** with modern styling.
- **Real-time predictions** with confidence levels.
- **Professional interface** with clear indicators.

## Requirements

```
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
```

## License

This project is created for educational purposes.


## Contributing

Contributions are welcome! Please follow clean code practices and provide detailed documentation.
