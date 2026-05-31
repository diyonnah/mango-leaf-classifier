from http.server import BaseHTTPRequestHandler, HTTPServer
import cgi
import io
import json
import os
import pickle
import urllib.parse

import joblib
import numpy as np
from PIL import Image
 

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
ALLOWED_PATHS = {"/", "/api/predict", "/predict"}


def _get_resample():
    return getattr(Image, "Resampling", Image).LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS


def extract_features(image):
    resample = _get_resample()
    image = image.convert("L")
    image = image.resize(IMAGE_SIZE, resample)
    arr = np.asarray(image, dtype=np.uint8)
    features = arr.flatten()
    return features


def _json_bytes(payload):
    return json.dumps(payload).encode("utf-8")


def _extract_file_from_form(form):
    for field_name in ("image", "file"):
        if field_name not in form:
            continue
        field = form[field_name]
        if isinstance(field, list):
            field = field[0] if field else None
        if field is not None and getattr(field, "file", None) is not None:
            return field
    return None


def _predict_from_bytes(image_bytes):
    if model is None:
        return {"error": "Model not loaded. Check your model file."}, 500

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
        "unhealthy": "Unhealthy",
    }

    try:
        numeric = int(prediction)
        result = label_map.get(numeric, str(prediction))
    except Exception:
        result = label_map.get(str(prediction).lower(), str(prediction))

    return {"result": result, "confidence": confidence}, 200


class handler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = _json_bytes(payload)
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        if status_code != 204:
            self.wfile.write(body)

    def _read_body(self):
        length = int(self.headers.get("content-length", "0"))
        return self.rfile.read(length) if length > 0 else b""

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path not in ALLOWED_PATHS:
            self._send_json(404, {"error": "Not found"})
            return

        self._send_json(200, {"status": "ok", "modelLoaded": model is not None})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path not in ALLOWED_PATHS:
            self._send_json(404, {"error": "Not found"})
            return

        content_type = self.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            self._send_json(400, {"error": "Expected multipart/form-data with an image field."})
            return

        form = cgi.FieldStorage(
            fp=io.BytesIO(self._read_body()),
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": content_type,
            },
            keep_blank_values=True,
        )

        file_field = _extract_file_from_form(form)
        if file_field is None:
            self._send_json(400, {"error": "No image uploaded. Use form field 'image' or 'file'."})
            return

        filename = getattr(file_field, "filename", "") or ""
        if filename == "":
            self._send_json(400, {"error": "No file selected."})
            return

        try:
            image_bytes = file_field.file.read()
            payload, status_code = _predict_from_bytes(image_bytes)
            self._send_json(status_code, payload)
        except Exception as error:
            self._send_json(500, {"error": f"Prediction failed: {str(error)}"})

    def log_message(self, format, *args):
        return


if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 5000), handler)
    print("Serving predict endpoint on http://127.0.0.1:5000/api/predict")
    server.serve_forever()
