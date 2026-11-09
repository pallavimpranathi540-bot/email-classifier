import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download required NLTK resource packages
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# 1. Load the Kaggle Dataset
# Kaggle's spam.csv uses 'latin-1' encoding and contains unneeded blank columns
print("Loading Kaggle dataset...")
df = pd.read_csv('spam.csv', encoding='latin-1')

# Clean columns: Keep only v1 (label) and v2 (text), then rename them
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print(f"Dataset successfully loaded! Total rows: {len(df)}")
print("Sample data from Kaggle:")
print(df.head(3))
print("-" * 50)

# 2. NLP Preprocessing Function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Tokenize words and convert to lowercase
    tokens = word_tokenize(text.lower())
    # Keep only alphanumeric words and remove stop words
    cleaned_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(cleaned_tokens)

print("Processing and cleaning text data using NLP pipeline...")
df['cleaned_message'] = df['message'].apply(clean_text)

# 3. Split data into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_message'], 
    df['label'], 
    test_size=0.2, 
    random_state=42
)

# 4. Feature Engineering (Text to Numbers using TF-IDF Vectorizer)
print("Converting text to mathematical vectors (TF-IDF)...")
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 5. Model Training (Naive Bayes Classifier)
print("Training the Machine Learning model...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# 6. Evaluation
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)

print("\n--- MODEL PERFORMANCE RESULTS ---")
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed Classification Report:")
print(classification_report(y_test, predictions))
print("-" * 50)

# 7. Test it with custom messages!
print("Testing the model with live custom text:")
custom_emails = [
    "Hey, are we still meeting up for dinner tonight at 7?",
    "URGENT! Your mobile number has won a £2,000 bonus prize! Call 09066364589 to claim now. Quick!"
]

custom_cleaned = [clean_text(email) for email in custom_emails]
custom_tfidf = vectorizer.transform(custom_cleaned)
custom_predictions = model.predict(custom_tfidf)

for email, prediction in zip(custom_emails, custom_predictions):
    print(f"\nMessage: '{email}'\nPrediction: [{prediction.upper()}]")
