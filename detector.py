import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# ── Load & label ──────────────────────────────────────────
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# ── Clean text ────────────────────────────────────────────
def clean_text(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'\[.*?\]', '', text)          # remove [bracketed content]
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)         # keep only letters
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]  # remove stopwords
    return ' '.join(tokens)

print("Cleaning text... (this may take a moment)")
df["clean_text"] = df["text"].apply(clean_text)
print("Done cleaning!")

# ── Split into train/test ─────────────────────────────────
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── TF-IDF Vectorization ──────────────────────────────────
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print(f"\nTF-IDF matrix shape: {X_train_tfidf.shape}")
print("\n✅ Preprocessing complete — ready for model training!")
from sklearn.linear_model import PassiveAggressiveClassifier
import pickle

# ── Train the model ───────────────────────────────────────
print("\nTraining model...")

model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

print("✅ Model trained!")

# ── Save model + vectorizer ───────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Model saved as model.pkl")
print("✅ Vectorizer saved as vectorizer.pkl")
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ── Evaluate ──────────────────────────────────────────────
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n📊 Accuracy: {accuracy * 100:.2f}%")

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
# ── Manual test ───────────────────────────────────────────
sample =  ["BREAKING: Scientists EXPOSE government LIES about climate change - they've been hiding the TRUTH for decades!! Share before they delete this!!"]
sample_clean = [clean_text(sample[0])]
sample_tfidf = vectorizer.transform(sample_clean)
prediction = model.predict(sample_tfidf)[0]

label = "🟢 REAL" if prediction == 1 else "🔴 FAKE"
print(f"\nSample prediction: {label}")