from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- Add this line
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # <-- Add this line

# Load the trained model and scaler
model = joblib.load('burnout_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Burnout Predictor API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.json

        # Extract features
        features = [
            data['Gender'],                 # 0 or 1
            data['Company Type'],          # 0 or 1
            data['WFH Setup Available'],   # 0 or 1
            data['Designation'],           # integer
            data['Resource Allocation'],   # float
            data['Mental Fatigue Score']   # float
        ]

        # Scale input
        features_scaled = scaler.transform([features])

        # Predict
        prediction = model.predict(features_scaled)[0]

        return jsonify({
            'burnout_risk': int(prediction),
            'message': "High Burnout" if prediction == 1 else "Low Burnout"
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)