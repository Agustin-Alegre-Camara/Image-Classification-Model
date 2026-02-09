import base64
import os
import json
import io
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from PIL import Image

# Paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "recognition_model.keras")
MODEL_PATH_LEGACY = os.path.join(MODEL_DIR, "recognition_model.h5")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

# Fallback class names, must match training order
DEFAULT_CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",   
    "Healthy",
    "Late_blight",
    "Leaf_mold",     
    "Mosaic_virus",
    "Septoria_leaf_spot",
    "Target_spot",
    "Two_spotted_spider_mite",
    "Yellow_leaf_curl_virus",
]

IMAGE_SIZE = 256

app = Flask(__name__)


def format_display_name(name):
    # Replace underscores with spaces for friendly display
    return name.replace("_", " ")


def load_class_names():
    if os.path.isfile(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            return json.load(f)
    return DEFAULT_CLASS_NAMES


def load_model():
    path = MODEL_PATH if os.path.isfile(MODEL_PATH) else MODEL_PATH_LEGACY
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH} or {MODEL_PATH_LEGACY}."
        )
    return tf.keras.models.load_model(path)


# Load once at startup
model = load_model()
class_names = load_class_names()


def predict_image(image_bytes):
    # Predict from image bytes. Returns label, confidence_str
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, 0)
    predictions = model.predict(img_array, verbose=0)
    idx = int(np.argmax(predictions[0]))
    label = class_names[idx]
    confidence = round(100 * float(np.max(predictions[0])), 2)
    return label, f"{confidence}%"


@app.route("/")
def index():
    return render_template(
        "index.html",
        classes=", ".join(format_display_name(c) for c in class_names),
        result_class="",
        prediction="",
        confidence="",
        confidence_pct="0",
        image_data="",
        image_type="jpeg",
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No image provided", 400
    file = request.files["image"]
    if file.filename == "":
        return "No image selected", 400
    try:
        data = file.read()
        label, confidence = predict_image(data)
        confidence_num = float(confidence.replace("%", ""))
        img = Image.open(io.BytesIO(data))
        image_format = (img.format or "JPEG").upper()
        mime_map = {"JPEG": "jpeg", "JPG": "jpeg", "PNG": "png", "GIF": "gif", "WEBP": "webp"}
        image_type = mime_map.get(image_format, "jpeg")
        image_data = base64.b64encode(data).decode("utf-8")
        return render_template(
            "index.html",
            classes=", ".join(format_display_name(c) for c in class_names),
            result_class="show",
            prediction=format_display_name(label),
            confidence=confidence,
            confidence_pct=str(int(confidence_num)),
            image_data=image_data,
            image_type=image_type,
        )
    except Exception as e:
        return f"Error: {e}", 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=False)
