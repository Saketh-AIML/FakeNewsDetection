from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return 'Fake News Detection API'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return jsonify({'prediction': prediction})
