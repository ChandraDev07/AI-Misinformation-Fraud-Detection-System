from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("fake_news_module/fake_news_model.pkl")
vectorizer = joblib.load("fake_news_module/vectorizer.pkl")

@app.route('/check_news', methods=['POST'])
def check_news():
    text = request.json['text']
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return jsonify({"fake": bool(pred==0)})

app.run(port=5000)
