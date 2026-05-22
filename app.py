from flask import Flask, request, jsonify, render_template
import pickle
import re

app = Flask(__name__)

with open("model/pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text or len(text.split()) < 5:
        return jsonify({"error": "Please enter a longer news text (min 5 words)."}), 400

    cleaned = clean_text(text)
    prediction = pipeline.predict([cleaned])[0]
    probabilities = pipeline.predict_proba([cleaned])[0]

    confidence = round(float(max(probabilities)) * 100, 2)
    label = "REAL" if prediction == 1 else "FAKE"

    return jsonify({
        "label": label,
        "confidence": confidence,
        "fake_prob": round(float(probabilities[0]) * 100, 2),
        "real_prob": round(float(probabilities[1]) * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)