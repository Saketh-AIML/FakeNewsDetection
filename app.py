from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

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

# Only needed for local testing
if __name__ == '__main__':
    app.run(debug=True)
