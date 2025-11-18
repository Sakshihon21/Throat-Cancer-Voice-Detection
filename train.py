import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from src.augment_and_features import extract_features
from src.model import build_model
import tensorflow as tf

DATA_DIR = "data"
LABELS = ["healthy", "cancer"]

X = []
y = []

print("📥 Loading audio files...")

for label_index, category in enumerate(LABELS):
    folder_path = os.path.join(DATA_DIR, category)
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            filepath = os.path.join(folder_path, file)
            features = extract_features(filepath)
            X.append(features)
            y.append(label_index)

X = np.array(X)
y = np.array(y)

print("🔄 Normalizing features...")
scaler = StandardScaler()
X = scaler.fit_transform(X)
dump(scaler, "models/scaler.joblib")

X = X[..., np.newaxis]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🚀 Building model...")
model = build_model((X_train.shape[1], 1))

callbacks = [
    tf.keras.callbacks.ModelCheckpoint("models/best_model.keras", save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]

print("🎯 Training...")
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=16,
    callbacks=callbacks
)

model.save("models/final_model.keras")
print("✅ Training complete! Model saved.")
