
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

fake["label"] = 0   
real["label"] = 1   

df = pd.concat([fake, real], ignore_index=True)


df["text"] = df["title"] + " " + df["text"]  
df = df[["text", "label"]].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df["text"]
y = df["label"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_df=0.7,          
        ngram_range=(1, 2),  
        max_features=50000
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=5,                 
        solver="lbfgs",
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)


y_pred = pipeline.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))


os.makedirs("model", exist_ok=True)
with open("model/pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\n Model saved to model/pipeline.pkl")
