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


def main():
    load_data()

if __name__ == "__main__":
    main()

