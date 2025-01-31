# app.py
from flask import Flask, request, jsonify, render_template
import pickle
import re
import string
import pandas as pd

app = Flask(__name__, static_folder='static')

# Load model and vectorizer
try:
    model = pickle.load(open('model.pkl', 'rb'))
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
except Exception as e:
    print(f"Error loading files: {str(e)}")
    model, vectorizer = None, None

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

import requests

# NEWS_API_KEY = "d4c96b43d3c04883a2790bd6c78d0117"  # Replace with your API key
# NEWS_API_URL = "https://newsapi.org/v2/top-headlines?country=us&apiKey=" + NEWS_API_KEY

@app.route('/get_news', methods=['GET'])
def get_news():
    try:
        response = requests.get(NEWS_API_URL)
        news_data = response.json()
        if news_data["status"] == "ok":
            articles = news_data.get("articles", [])
            return jsonify({"status": "success", "articles": articles})
        else:
            return jsonify({"status": "error", "message": "Failed to fetch news"})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error fetching news: {str(e)}"})


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or vectorizer is None:
            return jsonify({
                'status': 'error',
                'message': 'Model or vectorizer not loaded properly'
            })

        # Get text from request
        data = request.get_json()
        text = data['text']

        # Preprocess the text
        processed_text = wordopt(text)

        # Create DataFrame like in training
        testing_news = {"text": [processed_text]}
        new_def_test = pd.DataFrame(testing_news)

        # Transform the text using vectorizer
        new_xv_test = vectorizer.transform(new_def_test["text"])

        # Predict
        prediction = model.predict(new_xv_test)[0]

        # Get probability scores
        proba = model.predict_proba(new_xv_test)[0]
        confidence = float(max(proba) * 100)

        return jsonify({
            'status': 'success',
            'prediction': 'True News' if prediction == 1 else 'False News',  #here fake == true;
            'confidence': confidence,
            'text': text[:200] + '...' if len(text) > 200 else text
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error during prediction: {str(e)}'
        })

if __name__ == '__main__':
    app.run(debug=True)