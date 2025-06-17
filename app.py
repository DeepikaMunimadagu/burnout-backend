from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import sqlite3
import csv

app = Flask(__name__)
CORS(app)

model = joblib.load('burnout_model.pkl')
scaler = joblib.load('scaler.pkl')
DB_PATH = 'burnout.db'

@app.route('/')
def home():
    return "Burnout Predictor API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        features = [
            data['Gender'],
            data['Company Type'],
            data['WFH Setup Available'],
            data['Designation'],
            data['Resource Allocation'],
            data['Mental Fatigue Score']
        ]

        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]

        name = data.get("Name", "Anonymous")
        message = "High Burnout" if prediction == 1 else "Low Burnout"

        # Store in DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (name, gender, company_type, wfh, designation, resource_allocation, mental_fatigue_score, burnout_risk, message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, *features, int(prediction), message))
        conn.commit()
        conn.close()

        return jsonify({
            'burnout_risk': int(prediction),
            'message': message
        })

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/leaderboard', methods=['GET'])
def leaderboard():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, burnout_risk, message, timestamp FROM predictions ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()

    data = [
        {"name": row[0], "burnout_risk": row[1], "message": row[2], "timestamp": row[3]}
        for row in rows
    ]
    return jsonify(data)

@app.route('/download_csv', methods=['GET'])
def download_csv():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions")
    rows = cursor.fetchall()
    col_names = [description[0] for description in cursor.description]
    conn.close()

    csv_file = "burnout_predictions.csv"
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(col_names)
        writer.writerows(rows)

    return send_file(csv_file, as_attachment=True)

@app.route("/burnout_trends", methods=["GET"])
def burnout_trends():
    conn = sqlite3.connect("burnout.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DATE(timestamp), 
               SUM(burnout_risk = 1), 
               SUM(burnout_risk = 0) 
        FROM predictions 
        GROUP BY DATE(timestamp)
        ORDER BY DATE(timestamp)
    """)
    rows = cursor.fetchall()
    conn.close()

    trend_data = [{"date": r[0], "high": r[1], "low": r[2]} for r in rows]
    return jsonify(trend_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
