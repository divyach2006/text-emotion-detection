
from flask import Flask, render_template, request
import pickle
import re
import string

# Load saved model, vectorizer, and label encoder
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))
labelencoder = pickle.load(open("models/labelencoder.pkl", "rb"))

app = Flask(__name__)

# Clean text function
def clean_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = " ".join(text.split())
    return text

# Predict Emotion Function
def predict_emotion(text):
    cleaned = clean_text(text)
    x = vectorizer.transform([cleaned]).toarray()

    pred = model.predict(x)[0]        # numpy int64 -> 0/1/2

    pred = model.predict(x)[0]        # numpy int64 -> 0/1/2
    pred = int(pred)                  # convert to python int

    emotion = labelencoder.inverse_transform([pred])[0]
    return emotion

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        user_text = request.form["utext"]
        prediction = predict_emotion(user_text)
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":   # FIXED THIS LINE
    app.run(debug=True)