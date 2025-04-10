from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return 'Fake News Detection API is running.'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received JSON data:", data)

        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        vector = vectorizer.transform([text])
        prediction = model.predict(vector)[0]
        return jsonify({'prediction': prediction})

    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
