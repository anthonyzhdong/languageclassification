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