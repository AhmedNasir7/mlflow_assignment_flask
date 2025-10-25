import mlflow
import os

mlflow.set_tracking_uri("https://dagshub.com/username/repo.mlflow")   #tracking link here
os.environ["MLFLOW_TRACKING_USERNAME"] = "username"  #userame here
os.environ["MLFLOW_TRACKING_PASSWORD"] = "xyz"       #real token has to be here (i removed because of privacy)


from flask import Flask, request, render_template, jsonify
import mlflow.pyfunc
import numpy as np
from PIL import Image
import io

logged_model = "runs:/946f51dce94f4aa1a5eb4acc1b169447/pet_model"

try:
    Model = mlflow.pyfunc.load_model(logged_model)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading model:", e)
    Model = None

app = Flask(__name__)

def preprocess_uploaded_image(file):
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

@app.route("/")
def home():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    if Model is None:
        return jsonify({"error": "Model not loaded. Check your MLflow path."}), 500
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty file name"}), 400

    img_array = preprocess_uploaded_image(file)
    preds = Model.predict(img_array)
    probs = preds[0] if preds.ndim > 1 else [preds[0]]
    predicted_class = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return jsonify({
        "predicted_class": predicted_class,
        "confidence": round(confidence, 3)
    })

if __name__ == "__main__":
    app.run(debug=True)
