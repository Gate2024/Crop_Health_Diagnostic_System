# from flask import Flask, render_template, request, redirect, url_for, send_from_directory
# import tensorflow as tf
# import numpy as np
# import os
# import uuid
# import json
# import cv2

# # ================================
# # APP CONFIG
# # ================================
# app = Flask(__name__)

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease.keras")
# JSON_PATH = os.path.join(BASE_DIR, "models", "plant_disease.json")
# UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploading_images")

# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# if not os.path.exists(UPLOAD_FOLDER):
#     os.makedirs(UPLOAD_FOLDER)

# # ================================
# # LOAD MODEL
# # ================================
# print("Loading Model...")
# model = tf.keras.models.load_model(MODEL_PATH)
# print("Model Loaded Successfully ✅")

# # ================================
# # LOAD JSON
# # ================================
# with open(JSON_PATH, "r") as file:
#     plant_disease = json.load(file)

# print("JSON Loaded Successfully ✅")

# # ================================
# # LEAF VALIDATION USING OPENCV
# # ================================
# def is_leaf(image_path):
#     img = cv2.imread(image_path)

#     if img is None:
#         return False

#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # Green color range in HSV
#     lower_green = np.array([25, 40, 40])
#     upper_green = np.array([90, 255, 255])

#     mask = cv2.inRange(hsv, lower_green, upper_green)
#     green_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])

#     # If less than 8% green pixels → reject
#     if green_ratio < 0.08:
#         return False

#     return True

# # ================================
# # IMAGE PREPROCESSING
# # ================================
# def preprocess_image(image_path):
#     img = tf.keras.utils.load_img(image_path, target_size=(160, 160))
#     img_array = tf.keras.utils.img_to_array(img)
#     img_array = img_array / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# # ================================
# # PREDICTION FUNCTION
# # ================================
# def predict_disease(image_path):

#     # Step 1: Validate Leaf
#     if not is_leaf(image_path):
#         return {
#             "name": "Invalid Image",
#             "cause": "The uploaded image does not appear to be a plant leaf.",
#             "cure": "Please upload a clear green plant leaf image.",
#             "confidence": 0
#         }

#     # Step 2: Predict
#     img = preprocess_image(image_path)
#     prediction = model.predict(img)[0]

#     predicted_index = int(np.argmax(prediction))
#     confidence = round(float(prediction[predicted_index] * 100), 2)

#     disease_info = plant_disease[predicted_index]

#     return {
#         "name": disease_info["name"],
#         "cause": disease_info["cause"],
#         "cure": disease_info["cure"],
#         "confidence": confidence
#     }

# # ================================
# # ROUTES
# # ================================

# @app.route("/")
# def home():
#     return render_template("home.html")

# @app.route("/predict")
# def predict_page():
#     return render_template("predict.html")

# @app.route("/upload/", methods=["POST"])
# def upload():
#     if "img" not in request.files:
#         return redirect(url_for("predict_page"))

#     file = request.files["img"]

#     if file.filename == "":
#         return redirect(url_for("predict_page"))

#     unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
#     save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
#     file.save(save_path)

#     prediction = predict_disease(save_path)

#     return render_template(
#         "predict.html",
#         result=True,
#         imagepath=url_for("uploaded_file", filename=unique_filename),
#         prediction=prediction
#     )

# @app.route("/uploading_images/<filename>")
# def uploaded_file(filename):
#     return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# # ================================
# # RUN SERVER
# # ================================
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import tensorflow as tf
import numpy as np
import os
import uuid
import json
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

# ================================
# APP CONFIG
# ================================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "plant_disease.keras")
JSON_PATH = os.path.join(BASE_DIR, "models", "plant_disease.json")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploading_images")

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ================================
# LOAD MODEL
# ================================
print("Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully ✅")

# ================================
# LOAD JSON
# ================================
with open(JSON_PATH) as f:
    plant_disease_list = json.load(f)

# Convert list → dictionary (for fast lookup)
plant_disease = {item["name"]: item for item in plant_disease_list}

# IMPORTANT: Match model class order
class_names = sorted([item["name"] for item in plant_disease_list])

print("JSON Loaded Successfully ✅")
print("JSON Loaded Successfully ✅")

# ================================
# LEAF VALIDATION USING OPENCV
# ================================
def is_leaf(image_path):
    img = cv2.imread(image_path)

    if img is None:
        return False

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = np.sum(mask > 0) / (img.shape[0] * img.shape[1])

    if green_ratio < 0.08:
        return False

    return True

# ================================
# IMAGE PREPROCESSING (IMPORTANT FIX)
# ================================
def preprocess_image(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    # 🔥 MATCH TRAINING PREPROCESSING
    img_array = preprocess_input(img_array)

    return img_array

# ================================
# PREDICTION FUNCTION
# ================================
def predict_disease(image_path):

    # Step 1: Leaf validation
    if not is_leaf(image_path):
        return {
            "name": "Invalid Image",
            "cause": "The uploaded image does not appear to be a plant leaf.",
            "cure": "Please upload a clear green plant leaf image.",
            "confidence": 0,
            "fertilizer": "Not Applicable",
            "products": {}
        }

    # Step 2: Preprocess
    img = preprocess_image(image_path)

    # Step 3: Model Prediction
    predictions = model.predict(img)[0]
    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index]) * 100

    # Step 4: Get correct class name
    predicted_class_name = class_names[predicted_index]

    # Step 5: Get JSON info
    disease_info = plant_disease.get(predicted_class_name)

    if not disease_info:
        return {
            "name": "Prediction Error",
            "cause": "Class not found in JSON.",
            "cure": "Check model and JSON mapping.",
            "confidence": 0,
            "fertilizer": "Not Available",
            "products": {}
        }

    return {
        
    "name": disease_info["name"],
    "cause": disease_info["cause"],
    "cure": disease_info["cure"],
    "confidence": round(confidence, 2),
    "translations": disease_info.get("translations", {}),
    "fertilizer": disease_info.get("fertilizer", "Consult Agronomist"),
        "products": disease_info.get("products", {})
    }

# For Language Translation 
def get_language():
    lang = request.args.get("lang", "en")
    if lang not in ["en", "hi", "mr"]:
        return "en"
    return lang    
# ================================
# ROUTES
# ================================

# ================================
# ROUTES
# ================================

# Helper function to detect language
def get_language():
    lang = request.args.get("lang", "en")
    if lang not in ["en", "hi", "mr"]:
        lang = "en"
    return lang


@app.route("/")
def home():
    lang = get_language()
    return render_template("home.html", lang=lang)


@app.route("/predict")
def predict_page():
    lang = get_language()
    return render_template("predict.html", lang=lang)


@app.route("/upload/", methods=["POST"])
def upload():

    lang = get_language()   # 🔥 Get language first

    if "img" not in request.files:
        return redirect(url_for("predict_page", lang=lang))

    file = request.files["img"]

    if file.filename == "":
        return redirect(url_for("predict_page", lang=lang))

    unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
    file.save(save_path)

    prediction = predict_disease(save_path)

    return render_template(
        "predict.html",
        result=True,
        imagepath=url_for("uploaded_file", filename=unique_filename),
        prediction=prediction,
        lang=lang   # 🔥 VERY IMPORTANT
    )


@app.route("/uploading_images/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ================================
# RUN SERVER
# ================================
if __name__ == "__main__":
    app.run(debug=True)