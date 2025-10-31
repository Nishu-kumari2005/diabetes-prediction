from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, "model.pkl"))
scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))


FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]

@app.route("/")
def home():
    return "✅ Diabetes Prediction API is running! Use POST /predict"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        values = [data.get(f, 0) for f in FEATURES]
        
        arr = np.array(values).reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        
        prob = model.predict_proba(arr_scaled)[0][1]
        prediction = int(prob >= 0.5)

        return jsonify({
            "prediction": prediction,
            "probability": float(prob)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
