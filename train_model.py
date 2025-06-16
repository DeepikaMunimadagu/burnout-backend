import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
df = pd.read_csv("burnout.csv")  # Replace with your actual filename

# Select relevant features and target
X = df[['mental_fatigue_score', 'sleep_hours', 'work_hours']]
y = df['burnout']

# Preprocess
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "burnout_model.pkl")
joblib.dump(scaler, "scaler.pkl")
