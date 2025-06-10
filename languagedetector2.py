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

def load_data(dataset):
    try:
        df  = pd.read_csv(dataset)
        print("file load success")
        return df
    except FileNotFoundError:
        print("file load unsuccessful")
        return None
    

def load_all_datasets():
    train_df = load_data("train.csv")
    test_df = load_data("test.csv")
    valid_df = load_data("valid.csv")

    train_df = train_df.rename(columns={'text':'Text', 'labels':'language'})
    test_df = test_df.rename(columns={'text':'Text', 'labels':'language'})
    valid_df = valid_df.rename(columns={'text':'Text', 'labels':'language'})

    return train_df, test_df, valid_df


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

    print(f"Accuracy: {accuracy:.4f}")

    print(f"\n📊 DETAILED RESULTS:")
    print(classification_report(y_test, y_pred))

    # support -> amount of test data
    # precision -> out fo all of predictions, how many were actually correct
    # recall -> correctly identify as correct language
    # f1-score -> result that balances precision & recall

    return y_pred, y_prediction_probability

def create_language_detector(model, vectorizer):
    # wraps model & vectorizer into a function that will return the language it detects from text

    def detect_language(text, show_confidence=True, top_n=3):
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


def save_model(model, vectorizer, filename = 'languagemodel2.pkl'):
    from sklearn.pipeline import Pipeline

    complete_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', model)
    ])

    joblib.dump(complete_pipeline, filename)

    file_size = os.path.getsize(filename) / (1024 * 1024) #convert to mb

    return filename

def load_model(filename='languagemodel2.pkl'):
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

def trainmodel():

    train_df, test_df, valid_df = load_all_datasets()

    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)
    valid_df = preprocess_data(valid_df)

    # Prepare training data
    X_train = train_df['Text']
    y_train = train_df['language']
    
    # Prepare validation data
    X_valid = valid_df['Text']
    y_valid = valid_df['language']
    
    # Prepare test data
    X_test = test_df['Text']
    y_test = test_df['language']

    X_train_features, X_valid_features, vectorizer = create_features(X_train, X_valid)
    X_test_features = vectorizer.transform(X_test)

    model = train_model(X_train_features, y_train)

    y_pred, y_pred_proba = make_predictions(model, X_test_features, y_test)

    language_detector = create_language_detector(model, vectorizer)

    model_file = save_model(model, vectorizer)

    return language_detector, model_file

def test_detector():
    """Test the trained detector with sample texts"""
    print("\n🧪 Testing Language Detector")
    print("=" * 30)
    
    # Load the saved model
    pipeline = load_model()
    if pipeline is None:
        print("No trained model found. Please run train_full_pipeline() first.")
        return
    
    # Test samples
    test_texts = [
         "Hello, how are you today?",
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is revolutionizing technology across industries.",
            "Yesterday I went to the store to buy groceries for dinner.",
            "The weather forecast predicts rain throughout the weekend.",
            "Education is the most powerful weapon which you can use to change the world.",
            "Technology has transformed the way we communicate with each other.",
            "The scientific method involves observation, hypothesis, and experimentation.",
            "Hola, ¿cómo estás hoy?",
            "El zorro marrón rápido salta sobre el perro perezoso.",
            "La inteligencia artificial está transformando nuestras vidas.",
            "Ayer fui al mercado para comprar verduras frescas.",
            "Me gusta leer libros de historia en mi tiempo libre.",
            "La educación es fundamental para el desarrollo de una sociedad.",
            "El cambio climático es uno de los desafíos más importantes de nuestra época.",
            "La literatura española tiene una rica tradición que se remonta a siglos atrás.",
            "Bonjour, comment allez-vous aujourd'hui?",
            "Le renard brun rapide saute par-dessus le chien paresseux.",
            "L'intelligence artificielle révolutionne notre façon de travailler.",
            "Hier, je suis allé au marché pour acheter des légumes frais.",
            "J'aime beaucoup lire des romans français le soir.",
            "L'éducation est la clé du développement personnel et professionnel.",
            "La cuisine française est reconnue dans le monde entier pour sa sophistication.",
            "Les innovations technologiques transforment rapidement notre société moderne.",
                        "こんにちは、今日はいかがですか？",
            "素早い茶色のキツネが怠惰な犬を飛び越えます。",
            "人工知能は私たちの働き方を革命的に変えています。",
            "昨日市場に新鮮な野菜を買いに行きました。",
            "空いた時間に日本文学を読むのがとても好きです。",
            "教育は個人と社会の発展にとって不可欠です。",
            "日本語は独特な文字体系と豊かな表現力を持つ言語です。",
            "技術革新は現代社会を急速に変革しています。",
                        "สวัสดี วันนี้เป็นอย่างไรบ้าง?",
            "จิ้งจอกสีน้ำตาลที่รวดเร็วกระโดดข้ามสุนัขที่ขี้เกียจ",
            "ปัญญาประดิษฐ์กำลังปฏิวัติวิธีการทำงานของเรา",
            "เมื่อวานนี้ฉันไปตลาดเพื่อซื้อผักสด",
            "ฉันชอบอ่านวรรณกรรมไทยในเวลาว่าง",
            "การศึกษาเป็นกุญแจสำคัญสู่การพัฒนาส่วนบุคคลและสังคม",
            "ภาษาไทยมีประวัติศาสตร์ด้านวรรณกรรมที่ยาวนานและไวยากรณ์ที่ซับซ้อน",
            "นวัตกรรมทางเทคโนโลยีกำลังเปลี่ยนแปลงสังคมสมัยใหม่ของเราอย่างรวดเร็ว"
    ]
    
    for text in test_texts:
        prediction = pipeline.predict([text])[0]
        probabilities = pipeline.predict_proba([text])[0]
        confidence = probabilities.max()
        print(f"'{text}' -> {prediction} (confidence: {confidence:.3f})")


def main():

    # train the model (remove the #)
    #trainmodel()
    
    #language = quick_detect("languagemodel.pkl","hello this is a new language please detect what language this is please hello test")
    #print(language)
    test_detector()


if __name__ == "__main__":
    main()