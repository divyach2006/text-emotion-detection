import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
import os 

# Load the dataset
data = pd.read_csv("../data/Emotion_classify_Data.csv")

print("Dataset loaded successfully!")
print(data.head())

# Clean text function
def clean_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = " ".join(text.split())
    return text

# Apply cleaning
data['Cleaned_Comment'] = data['Comment'].apply(clean_text)
print("\nSample cleaned text:")
print(data['Cleaned_Comment'].head())
print(data['Emotion'].value_counts())

# Step 1 — Convert text to numbers
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data['Cleaned_Comment']).toarray()

# Step 2 — Encode emotions
le = LabelEncoder()
y = le.fit_transform(data['Emotion'])

# Step 3 — Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data Prepared Successfully!")

# Step 4 — Train model
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

print("Training model...")
model = LinearSVC()
model.fit(X_train, y_train)
print("Model trained successfully!")

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Step 5 — Save model, vectorizer, and label encoder
os.makedirs("models", exist_ok=True)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("models/labelencoder.pkl", "wb") as f:
    pickle.dump(le, f)        # <<< FIXED HERE

print("All model files saved successfully!")
