from flask import Flask, request, jsonify
import numpy as np
import os
import sys
from tensorflow.keras.models import load_model

sys.path.append(os.path.join(os.path.dirname(__file__), '../src')) 

app = Flask(__name__)

MODEL_FILE_NAME = "lstm_bridge_health.h5"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", MODEL_FILE_NAME)

try:
    model = load_model(MODEL_PATH)
    print(f"[SUCCESS] LSTM Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Could not load model from {MODEL_PATH}")
    print(f"Error details: {e}")
    model = None

@app.route('/')
def home():
    if model is not None:
        return "Bridge Health Monitoring API is Running and Model is Loaded ✅"
    else:
        return "Bridge Health Monitoring API is Running, but Model Loading Failed ⚠️"


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Check server logs."}), 503
        
    try:
        data = request.get_json()
        if not data or 'vibration_data' not in data:
            return jsonify({"error": "No 'vibration_data' provided in the request body."}), 400

        input_data = np.array(data['vibration_data'])
        
        if input_data.shape[0] < 200:
             return jsonify({"error": "Vibration data must contain at least 200 points."}), 400

        input_data = input_data[:200]
        input_data = input_data.reshape((1, 200, 1))

        prediction = model.predict(input_data, verbose=0)[0][0]
        
        result = "Damaged ⚠️" if prediction > 0.5 else "Healthy ✅"
        damage_confidence = float(prediction)
        health_index = round(100 * (1 - damage_confidence), 2)


        return jsonify({
            "status": "Success",
            "bridge_condition": result,
            "health_index": health_index,
            "damage_confidence": damage_confidence
        })

    except Exception as e:
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)