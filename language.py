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

class LanguageDetector:  # Fixed class name (should be CapitalCase)
    def __init__(self, max_features=10000, ngram_range=(1,4), test_size=0.2, random_state=42):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.test_size = test_size
        self.random_state = random_state
        
        # Initialize components that will be created during training
        self.pipeline = None
        self.vectorizer = None
        self.classifier = None
        self.classes_ = None
        
        # Performance metrics
        self.train_accuracy = None
        self.test_accuracy = None

    def load_data(self, file_path='dataset.csv'):  # Fixed: Made this a method of the class
        """Load dataset from CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"✅ File loaded successfully: {len(df)} samples")
            return df
        except FileNotFoundError:
            print("❌ File load was unsuccessful - dataset.csv not found")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            raise

    def clean_text(self, text):  # Fixed: Made this a method of the class
        """Clean individual text strings"""
        # checks if text is missing
        if pd.isna(text):
            return ""
        
        text = str(text)
        # removes anything more than 1 white space
        text = re.sub(r'\s+', ' ', text.strip())
        return text

    def preprocess_data(self, df):  # Fixed: Made this a method of the class
        """Preprocess the entire dataset"""
        print("🧹 Cleaning data...")
        original_count = len(df)
        
        # apply clean_text to each text field
        df['Text'] = df['Text'].apply(self.clean_text)  # Fixed: use self.clean_text
        
        # Remove empty texts
        df = df[df['Text'].str.len() > 0]
        
        removed = original_count - len(df)
        if removed > 0:
            print(f"   Removed {removed} empty samples")
        
        return df

    def split_data(self, df):  # Fixed: Corrected method signature
        """Split data into training and testing sets"""
        print("📊 Splitting data...")
        
        X = df['Text']
        y = df['language']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y  # Ensures balanced representation of all languages
        )
        
        print(f"   Training: {len(X_train)} samples")
        print(f"   Testing: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def create_pipeline(self):  # Fixed: Corrected method signature
        """Create the ML pipeline"""
        vectorizer = TfidfVectorizer(
            analyzer='char',                # Character-level analysis
            ngram_range=self.ngram_range,   # Range of n-gram sizes (1-4 characters)
            max_features=self.max_features, # Limit features to prevent overfitting
            lowercase=True,                 # Normalize case
            strip_accents='unicode'         # Handle accented characters properly
        )
        
        # Logistic regression classifier
        classifier = LogisticRegression(
            max_iter=1000,          # Sufficient iterations for convergence
            random_state=self.random_state,
            solver='liblinear',     # Efficient for smaller datasets
            multi_class='ovr',      # One-vs-Rest strategy for multiclass
            C=1.0                   # Regularization strength (1/lambda)
        )
        
        # Combine into pipeline
        pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        
        return pipeline

    def train_model(self, X_train, y_train):  # Fixed: Corrected method signature
        """Train the language detection model"""
        print("🧠 Training model...")
        
        self.pipeline = self.create_pipeline()  # Fixed: was self.create.pipeline()
        self.pipeline.fit(X_train, y_train)

        self.vectorizer = self.pipeline.named_steps['vectorizer']
        self.classifier = self.pipeline.named_steps['classifier']  # Fixed: was 'classifer'
        self.classes_ = self.classifier.classes_
        
        print(f"✅ Training completed: {len(self.classes_)} languages learned")

    def evaluate_model(self, X_train, X_test, y_train, y_test):  # Fixed: Corrected method signature
        """Evaluate model performance"""
        print("📈 Evaluating performance...")
        
        self.train_accuracy = self.pipeline.score(X_train, y_train)
        self.test_accuracy = self.pipeline.score(X_test, y_test)

        print(f"   Training accuracy: {self.train_accuracy:.4f}")
        print(f"   Test accuracy: {self.test_accuracy:.4f}")
        print(f"   Generalization gap: {self.train_accuracy - self.test_accuracy:.4f}")

        # Generate detailed classification report
        y_pred = self.pipeline.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy,
            'classification_report': report
        }

    def analyze_learned_patterns(self, top_n=5):  # Fixed: Corrected method signature
        """Analyze what patterns the model learned"""
        if self.vectorizer is None or self.classifier is None:
            print("❌ Model must be trained first")
            return
            
        feature_names = self.vectorizer.get_feature_names_out()
        weights = self.classifier.coef_

        print(f"\n🔍 Top {top_n} patterns per language:")
        print("-" * 40)

        # Show first 5 languages to avoid too much output
        for i, language in enumerate(self.classes_[:5]):
            lang_weights = weights[i]
            top_indices = np.argsort(lang_weights)[-top_n:][::-1]

            print(f"\n{language}:")  # Fixed: removed incorrect syntax
            for idx in top_indices:
                feature = feature_names[idx]
                weight = lang_weights[idx]
                print(f"   '{feature}': {weight:.3f}")

    def create_predictor(self):  # Fixed: Corrected method signature
        """Create a prediction function"""
        if self.pipeline is None:  # Fixed: check pipeline instead of vectorizer/classifier
            raise ValueError("Model must be trained first")

        def predict_language(text, return_probabilities=False):
            if not text or len(text.strip()) < 3:
                return {"error": "Text too short (minimum 3 characters)"}  # Fixed: return dict
                
            try:
                prediction = self.pipeline.predict([text])[0]
                probabilities = self.pipeline.predict_proba([text])[0]
                confidence = probabilities.max()

                result = {
                    'predicted_language': prediction,
                    'confidence': confidence
                }
                
                if return_probabilities:
                    prob_dict = dict(zip(self.classes_, probabilities))
                    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
                    result['all_probabilities'] = sorted_probs[:3]

                return result
                
            except Exception as e:
                return {"error": f"Prediction failed: {str(e)}"}  # Fixed: return dict
                
        return predict_language

    def save_model(self, file_path='language_detector.pkl'):  # Fixed: method name and signature
        """Save the trained model"""
        if self.pipeline is None:
            raise ValueError("No trained model to save")
            
        joblib.dump(self.pipeline, file_path)
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"💾 Model saved: {file_path} ({file_size:.2f} MB)")

    @classmethod  # Fixed: Added missing decorator
    def load_model(cls, file_path='language_detector.pkl'):
        """Load a previously saved model"""
        try:
            pipeline = joblib.load(file_path)
            
            # Create detector instance and populate with loaded components
            detector = cls()
            detector.pipeline = pipeline
            detector.vectorizer = pipeline.named_steps['vectorizer']
            detector.classifier = pipeline.named_steps['classifier']
            detector.classes_ = detector.classifier.classes_
            
            print(f"📂 Model loaded: {file_path}")
            return detector
            
        except FileNotFoundError:
            print(f"❌ Model file not found: {file_path}")
            raise

def train_language_detector(dataset_path='dataset.csv'):
    """Main training function"""
    print("🚀 Language Detection Training Pipeline")
    print("=" * 50)
    
    # Initialize detector
    detector = LanguageDetector()  # Fixed: use correct class name
    
    # Load and preprocess data
    df = detector.load_data(dataset_path)  # Fixed: pass file path
    df = detector.preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = detector.split_data(df)
    
    # Train model
    detector.train_model(X_train, y_train)
    
    # Evaluate performance
    metrics = detector.evaluate_model(X_train, X_test, y_train, y_test)
    
    # Analyze learned patterns
    detector.analyze_learned_patterns()
    
    # Create prediction function
    predictor = detector.create_predictor()
    
    # Save model
    detector.save_model()
    
    print(f"\n✅ Training complete! Test accuracy: {detector.test_accuracy:.3f}")
    
    return detector, predictor

def test_predictions(predictor):
    """Test the trained model with sample texts"""
    test_examples = [
        "Hello, how are you doing today?",
        "Bonjour, comment allez-vous?",
        "Hola, ¿cómo estás hoy?",
        "Guten Tag, wie geht es Ihnen?",
        "Ciao, come stai oggi?",
    ]
    
    print(f"\n🧪 Testing predictions:")
    print("-" * 30)
    
    for text in test_examples:
        result = predictor(text, return_probabilities=True)
        
        if 'error' in result:
            print(f"❌ '{text[:30]}...' → {result['error']}")
        else:
            lang = result['predicted_language']
            conf = result['confidence']
            
            print(f"📝 '{text[:30]}...'")
            print(f"   → {lang} ({conf:.3f})")
            
            if 'all_probabilities' in result:
                top_3 = result['all_probabilities']
                print(f"   Top 3: {[(l, f'{p:.3f}') for l, p in top_3]}")
            print()

def main():
    """Main execution function"""
    try:
        # Train the model
        detector, predictor = train_language_detector()
        
        # Test with examples
        test_predictions(predictor)
        
        print("🎯 Language detection system ready!")
        print("Use predictor('your text') to classify new text.")
        
        return detector, predictor
        
    except Exception as e:
        print(f"❌ Error in main execution: {str(e)}")
        return None, None

if __name__ == "__main__":
    # Run the complete system
    detector, predictor = main()