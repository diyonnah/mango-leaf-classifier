from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import pickle
import numpy as np
from PIL import Image
import os
import io

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Load your trained SVM model ---
# Look for known filenames first, then fall back to any .sav/.pkl in model/ or root
DEFAULT_NAMES = ["svm_model.pkl", "svm_model.joblib", "MANGO_LEAF_Classifier.sav", "MANGO_LEAF_Classifier.pkl"]
MODEL_PATH = None

search_dirs = []
if os.path.isdir("model"):
    search_dirs.append("model")
search_dirs.append(APP_ROOT)

for directory in search_dirs:
    for name in DEFAULT_NAMES:
        candidate = os.path.join(directory, name)
        if os.path.exists(candidate):
            MODEL_PATH = candidate
            break
    if MODEL_PATH is not None:
        break

# fallback: pick first .sav/.pkl/.joblib file in search dirs
if MODEL_PATH is None:
    for directory in search_dirs:
        for fname in os.listdir(directory):
            if fname.lower().endswith((".sav", ".pkl", ".joblib")):
                MODEL_PATH = os.path.join(directory, fname)
                break
        if MODEL_PATH is not None:
            break

model = None
if MODEL_PATH is None:
    print("⚠️  Model file not found in model/ folder. Place your trained model there (e.g. MANGO_LEAF_Classifier.sav).")
else:
    # try joblib first, then pickle
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded successfully from {MODEL_PATH} (joblib)")
    except Exception:
        try:
            with open(MODEL_PATH, "rb") as mf:
                model = pickle.load(mf)
            print(f"✅ Model loaded successfully from {MODEL_PATH} (pickle)")
        except Exception as e:
            print(f"⚠️  Failed to load model from {MODEL_PATH}: {e}")


# --- Feature extraction ---
# ⚠️ IMPORTANT: This must match EXACTLY how you extracted features during training.
# The saved dataset shows 50x50 grayscale flattened images (dtype=uint8)
IMAGE_SIZE = (50, 50)

def _get_resample():
    # Pillow compatibility for resampling filter
    return getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS

def extract_features(image):
    """
    Preprocess image to match training pipeline: convert to grayscale, resize to 50x50,
    flatten to 1D uint8 array. Do NOT normalize if model was trained on raw uint8 values.
    """
    resample = _get_resample()
    image = image.convert("L")
    image = image.resize(IMAGE_SIZE, resample)
    arr = np.asarray(image, dtype=np.uint8)
    features = arr.flatten()
    return features


@app.route("/")
def home():
    # serve index.html from project root so a missing templates/ folder won't break
    return send_from_directory(APP_ROOT, "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Check your model file."}), 500

    # accept either 'image' or 'file' form field names
    file = request.files.get("image") or request.files.get("file")
    if file is None:
        return jsonify({"error": "No image uploaded. Use form field 'image' or 'file'."}), 400

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        # Read and process the image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # Extract features and ensure shape (1, -1)
        features = extract_features(image)
        features = np.asarray(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        # Get confidence score if available
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = round(float(max(proba)) * 100, 2)
        elif hasattr(model, "decision_function"):
            score = model.decision_function(features)[0]
            confidence = round(min(abs(float(score)) * 10, 100), 2)

        # Map prediction to human labels (training used 0/1)
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

        return jsonify({
            "result": result,
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
