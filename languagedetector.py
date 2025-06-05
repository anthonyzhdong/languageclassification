import pandas as pd           # For handling our dataset (like Excel but in Python)
import numpy as np           # For mathematical operations on arrays
import re                    # For cleaning text (removing extra spaces, etc.)

# Scikit-learn imports - the machine learning library
from sklearn.model_selection import train_test_split     # Split data into train/test
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numbers
from sklearn.linear_model import LogisticRegression     # Our main algorithm
from sklearn.metrics import accuracy_score, classification_report  # Measure performance
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt  # For creating graphs
import joblib                   # For saving our trained model
import os

def load_data():
    try:
        df  = pd.read_csv("dataset.csv")
        print("file load success")
        return df
    except FileNotFoundError:
        print("file load unsuccessful")
        return None

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'\s+', ' ', text.strip())

    return text

def preprocess_data(df):
    df['Text'] = df['Text'].apply(clean_text)
    df = df[df['Text'].str.len() > 0]

    return df

def create_features(X_train, X_test):
    # TF - count how often each ngram appears in a text
    # IDF - gives higher weight to n-grams that are rare across languages
    # features that are distinctive for each language get higher scores

    vectorizer = TfidfVectorizer(
        analyzer = 'char', # Character level
        ngram_range = (1,4), # 1-4 character combinations
        max_features = 10000, # limit to 10k most important character combinations
        lowercase = True, # normalise casing
        strip_accents = 'unicode' # removes accents on characters
    )
    # fit() scans through all ngrams and lists then -> attached to index
    # transform() converts texts to numbers
    # -> each position represents a ngram for the vocabulary
    # -> value at that position is the TF-IDF score
    X_train_features = vectorizer.fit_transform(X_train)
    X_test_features = vectorizer.transform(X_test)

    return X_train_features, X_test_features, vectorizer


# Training the model -> find the best weights that minimize prediction errors
# process -> start with random weights -> make predictions on training data -> calculate how wrong we are (loss function)
# adjust weights to reduce errors & repeat until convergence
def train_model(X_train_features, y_train):
    model = LogisticRegression(
        max_iter=1000, # maximum iterations
        random_state=42, # reproducible results
        solver='liblinear', # good for smaller datasets
        multi_class='ovr', # one-vs-rest for multiclass
        C=1.0 # regularization parameter
    )
    model.fit(X_train_features,y_train)

    return model


def make_predictions(model, X_test_features, y_test):
    # using trained model to actually make predictions

    # get predictions
    y_pred = model.predict(X_test_features)

    # get prediction probabilities
    y_prediction_probability = model.predict_proba(X_test_features)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📊 DETAILED RESULTS:")
    print(classification_report(y_test, y_pred))

    # support -> amount of test data
    # precision -> out fo all of predictions, how many were actually correct
    # recall -> correctly identify as correct language
    # f1-score -> result that balances precision & recall

    return y_pred, y_prediction_probability

def split_data(df):
    X = df['Text']
    y = df['language']

    X_train, X_test, y_train, y_test = train_test_split(
        X,y, # data to split
        test_size = 0.2, # test size
        random_state = 42, # same split every time
        stratify=y # keep same % of each language in both sets
    )

    return X_train, X_test, y_train, y_test

def create_language_detector(model, vectorizer):
    # wraps model & vectorizer into a function that will return the language it detects from text

    def detect_language(text, show_confidence=True, top_n=1):
        #if not text or len(text.strip()) < 3:
        #return {"error"}
        
        # Clean the text
        clean_input = re.sub(r'\s+',' ',text.strip())
        # convert to features
        text_features = vectorizer.transform([clean_input])
        # make prediction
        prediction = model.predict(text_features)[0]
        # get confidence scores
        probabilities = model.predict_proba(text_features)[0]
        # get top N predictions
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        top_predictions = []

        for idx in top_indices:
            lang = model.classes_[idx]
            prob = probabilities[idx]
            top_predictions.append({
                'language': lang,
                'probability': prob,
                'percentage': f"{prob*100:.1f}%"
            })
            
        result = {
            'predicted_language': prediction,
            'confidence': probabilities.max(),
            'top_predictions': top_predictions
        }

        if show_confidence:
            print(f"Text: '{text[:50]}...'")
            print(f"Predicted: {prediction} ({probabilities.max():.3f})")
            print(f"Top {top_n}: {[p['language'] + f' ({p['percentage']})' for p in top_predictions]}")

        return result
    return detect_language


def save_model(model, vectorizer, filename = 'languagemodel.pkl'):
    from sklearn.pipeline import Pipeline

    complete_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', model)
    ])

    joblib.dump(complete_pipeline, filename)

    file_size = os.path.getsize(filename) / (1024 * 1024) #convert to mb

    return filename

def load_model(filename='languagemodel.pkl'):
    try:
        pipeline = joblib.load(filename)
        return pipeline
    except FileNotFoundError:
        return None
    
def quick_detect(model, text):
    try: 
        pipeline = joblib.load(model)
        return pipeline.predict([text])[0]
    except:
        return "error"

def main():
    # training -> 
    #df = load_data()
    #df = preprocess_data(df)

    #X_train, X_test, y_train, y_test = split_data(df)
    #X_train_features, X_test_features, vectorizer = create_features(X_train, X_test)

    #model = train_model(X_train_features, y_train)

    #y_pred, y_pred_proba = make_predictions(model, X_test_features, y_test)

    #language_detector = create_language_detector(model, vectorizer)
    
    #model_file = save_model(model, vectorizer)
    language = quick_detect("languagemodel.pkl","hello this is a new language please detect what language this is please hello test")
    print(language)


if __name__ == "__main__":
    main()