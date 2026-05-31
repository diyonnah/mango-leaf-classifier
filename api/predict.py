from flask import Flask, request, jsonify
import joblib
import pickle
import numpy as np
from PIL import Image
import os
import io

app = Flask(__name__)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DEFAULT_NAMES = [
    "svm_model.pkl",
    "svm_model.joblib",
    "MANGO_LEAF_Classifier.sav",
    "MANGO_LEAF_Classifier.pkl"
]

MODEL_PATH = None
search_dirs = [os.path.join(ROOT_DIR, "model"), ROOT_DIR]

for directory in search_dirs:
    if not os.path.isdir(directory):
        continue
    for name in DEFAULT_NAMES:
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            MODEL_PATH = candidate
            break
    if MODEL_PATH is not None:
        break

if MODEL_PATH is None:
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        for fname in os.listdir(directory):
            if fname.lower().endswith((".sav", ".pkl", ".joblib")):
                MODEL_PATH = os.path.join(directory, fname)
                break
        if MODEL_PATH is not None:
            break

model = None
if MODEL_PATH is not None:
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as model_file:
            model = pickle.load(model_file)

IMAGE_SIZE = (50, 50)


def _get_resample():
    return getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS


def extract_features(image):
    resample = _get_resample()
    image = image.convert("L")
    image = image.resize(IMAGE_SIZE, resample)
    arr = np.asarray(image, dtype=np.uint8)
    features = arr.flatten()
    return features


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "modelLoaded": model is not None})


@app.route("/", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check your model file."}), 500

    file = request.files.get("image") or request.files.get("file")
    if file is None:
        return jsonify({"error": "No image uploaded. Use form field 'image' or 'file'."}), 400

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        features = extract_features(image)
        features = np.asarray(features).reshape(1, -1)
        prediction = model.predict(features)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = round(float(max(proba)) * 100, 2)
        elif hasattr(model, "decision_function"):
            score = model.decision_function(features)[0]
            confidence = round(min(abs(float(score)) * 10, 100), 2)

        label_map = {
            0: "Healthy",
            1: "Unhealthy",
            "dead": "Unhealthy",
            "alive": "Healthy",
            "healthy": "Healthy",
            "unhealthy": "Unhealthy"
        }

        try:
            numeric = int(prediction)
            result = label_map.get(numeric, str(prediction))
        except Exception:
            result = label_map.get(str(prediction).lower(), str(prediction))

        return jsonify({"result": result, "confidence": confidence})

    except Exception as error:
        return jsonify({"error": f"Prediction failed: {str(error)}"}), 500
