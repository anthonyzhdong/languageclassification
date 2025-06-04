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

def demonstrateNgrams():
    examples = {
        'English': "the quick brown",
        'Spanish': "el rápido marrón", 
        'French': "le rapide brun",
        'German': "der schnelle braun"
    }

    def extract_ngrams(text, n):
        text = text.replace(" ","")
        ngrams = []

        for i in range(len(text) - n + 1):
            ngram = text[i:i+n]
            ngrams.append(ngram)
        return ngrams
    
    for language, text in examples.items():
        trigrams = extract_ngrams(text, 3)
        print(trigrams)

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

def explain_logistic_regression():
    """
    Demonstrate the math behind logistic regression
    
    This helps understand what happens inside the algorithm
    """
    
    print("\n🧮 LOGISTIC REGRESSION MATHEMATICS")
    print("=" * 50)
    
    print("🎯 Goal: Convert any number to a probability (0 to 1)")
    print()
    
    # Demonstrate sigmoid function
    print("1️⃣  SIGMOID FUNCTION: σ(z) = 1 / (1 + e^(-z))")
    
    z_values = [-5, -2, -1, 0, 1, 2, 5]
    print("   Examples:")
    for z in z_values:
        sigmoid = 1 / (1 + np.exp(-z))
        print(f"      σ({z:2d}) = {sigmoid:.4f}")
    
    print(f"\n   Key insight: Sigmoid maps any number to (0,1) range!")
    
    # Demonstrate softmax for multiple classes
    print(f"\n2️⃣  SOFTMAX (for multiple languages):")
    print("   P(language k) = e^(z_k) / Σ(e^(z_j))")
    
    # Example with 3 languages
    z_scores = np.array([2.1, 1.3, 0.5])  # Linear scores for English, Spanish, French
    softmax_probs = np.exp(z_scores) / np.sum(np.exp(z_scores))
    
    languages = ['English', 'Spanish', 'French']
    print(f"\n   Example linear scores: {z_scores}")
    print(f"   Softmax probabilities:")
    for lang, prob in zip(languages, softmax_probs):
        print(f"      {lang}: {prob:.4f} ({prob*100:.1f}%)")
    print(f"   Sum: {np.sum(softmax_probs):.6f} (must equal 1.0)")
    
    print(f"\n3️⃣  TRAINING PROCESS:")
    print("   • Start with random weights")
    print("   • For each training example:")
    print("     - Calculate prediction")
    print("     - Compare with correct answer")
    print("     - Adjust weights to reduce error")
    print("   • Repeat until weights converge")


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
# Run the explanation
#explain_logistic_regression()

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

def analyze_model_weights(model, vectorizer):
    """
    Understand what patterns the model learned for each language
    
    This is the "explainable AI" part - we can see exactly why the model
    makes its decisions
    
    Args:
        model: Trained logistic regression model
        vectorizer: The TF-IDF vectorizer (to get feature names)
    """
    
    print("\n🔍 ANALYZING WHAT THE MODEL LEARNED")
    print("=" * 50)
    
    # Get feature names (the n-grams)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the weight matrix
    weights = model.coef_
    
    print(f"🧠 Model has learned {weights.shape[1]:,} feature weights for each language")
    print(f"📊 Weight matrix shape: {weights.shape}")
    
    # Analyze top features for each language
    print(f"\n🏷️  TOP DISCRIMINATIVE FEATURES:")
    
    # Show analysis for first 5 languages
    languages_to_show = min(5, len(model.classes_))
    
    for i in range(languages_to_show):
        language = model.classes_[i]
        lang_weights = weights[i]
        
        print(f"\n🌍 {language.upper()}:")
        
        # Find features with highest positive weights
        top_positive_idx = np.argsort(lang_weights)[-8:][::-1]
        print(f"   Most indicative n-grams (positive weights):")
        for idx in top_positive_idx:
            feature = feature_names[idx]
            weight = lang_weights[idx]
            print(f"     '{feature}': {weight:.3f}")
        
        # Find features with most negative weights
        top_negative_idx = np.argsort(lang_weights)[:5]
        print(f"   Least indicative n-grams (negative weights):")
        for idx in top_negative_idx:
            feature = feature_names[idx]
            weight = lang_weights[idx]
            print(f"     '{feature}': {weight:.3f}")
    
    print(f"\n💡 INTERPRETATION:")
    print("   • High positive weights = strong indicators FOR that language")
    print("   • High negative weights = strong indicators AGAINST that language")
    print("   • These patterns align with linguistic knowledge!")

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


df = load_data()
df = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(df)
X_train_features, X_test_features, vectorizer = create_features(X_train, X_test)

model = train_model(X_train_features, y_train)

y_pred, y_pred_proba = make_predictions(model, X_test_features, y_test)
#analyze_model_weights(model, vectorizer)

language_detector = create_language_detector(model, vectorizer)
# Test it with examples
print("\n🎯 TESTING OUR LANGUAGE DETECTOR")
print("=" * 45)

test_examples = [
    "Hello",
    "Bonjour, comment allez-vous?",
    "Hola",
    "Guten Tag, wie geht es Ihnen?",
    "こんにちは、元気ですか？"
]

for example in test_examples:
    print(f"\n🧪 Testing: '{example}'")
    result = language_detector(example)
    if 'error' not in result:
        print(f"   → {result['predicted_language']} (confidence: {result['confidence']:.3f})")

model_file = save_model(model, vectorizer)

#demonstrateNgrams()