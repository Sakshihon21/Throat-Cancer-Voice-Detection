import numpy as np
import librosa
from joblib import load
from tensorflow.keras.models import load_model
from src.augment_and_features import extract_features

scaler = load("models/scaler.joblib")
model = load_model("models/best_model.keras")

def predict_audio(file_path):
    features = extract_features(file_path)
    features = scaler.transform([features])
    features = features[..., np.newaxis]
    prediction = model.predict(features)[0][0]
    print("Cancer Probability:", prediction)
    print("Prediction:", "Cancer" if prediction > 0.5 else "Healthy")

# Example:
# predict_audio("test.wav")
