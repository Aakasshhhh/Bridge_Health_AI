import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "../data/vibrations/vibration_data.csv"
MODEL_PATH = "../models/lstm_bridge_health.h5"

print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded successfully with shape: {df.shape}")

X = df.drop("label", axis=1).values
y = df["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "scaler.pkl")
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
joblib.dump(scaler, SCALER_PATH)

X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("⚙️ Building LSTM model...")
model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("🚀 Training the model...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")

current_dir = os.path.dirname(os.path.abspath(__file__)) 

MODEL_PATH = os.path.join(current_dir, "..", "models", "lstm_bridge_health.h5")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

model.save(MODEL_PATH)

print(f"\n✅ LSTM Model saved successfully!")
print(f"File location confirmed: {MODEL_PATH}")