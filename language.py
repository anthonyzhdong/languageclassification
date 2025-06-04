import pandas as pd           # For handling our dataset (like Excel but in Python)
import numpy as np           # For mathematical operations on arrays
import re                    # For cleaning text (removing extra spaces, etc.)

# Scikit-learn imports - the machine learning library
from sklearn.model_selection import train_test_split     # Split data into train/test
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numbers
from sklearn.linear_model import LogisticRegression     # Our main algorithm
from sklearn.metrics import accuracy_score, classification_report  # Measure performance

import matplotlib.pyplot as plt  # For creating graphs
import joblib                   # For saving our trained model

def load_data():
    try:
        df = pd.read_csv('dataset.csv')
        print("File load was successful")
    except FileNotFoundError:
        print("File load was unsuccessful")
        return None
    
    print(df.shape)
    print(df.columns.tolist())
    print(df)

    return df

def clean_text(text):
    # checks if text is missing
    if pd.isna(text):
        return ""
    
    text = str(text)
    # removes anything more than 1 white space
    text = re.sub(r'\s+', ' ', text.strip())

    return text

def preprocess_data(df):
    # apply clean_text to each text field
    df['Text'] = df['Text'].apply(clean_text)

    df = df[df['Text'].str.len() > 0]

    return df
def main():
    df = load_data()
    df = preprocess_data(df)

if __name__ == "__main__":
    main()

