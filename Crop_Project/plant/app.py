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

from flask import Flask, render_template, request, redirect, url_for, session
import tensorflow as tf
import numpy as np
import os
import uuid
import json
import cv2
import sqlite3
from werkzeug.security import check_password_hash
from werkzeug.security import generate_password_hash
from tensorflow.keras.applications.efficientnet import preprocess_input
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
# ===============================
# CREATE FLASK APP FIRST
# ===============================
app = Flask(__name__)

# ✅ Secret key MUST be after app creation
app.secret_key = "crop_health_secret_key"

# ================= EMAIL CONFIGURATION =================

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'narayanadhude01@gmail.com'
app.config['MAIL_PASSWORD'] = 'oinbeyzgufdrqzqy'

mail = Mail(app)

serializer = URLSafeTimedSerializer(app.secret_key)

# =======================================================

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


def init_db():
    conn = sqlite3.connect("users.db", timeout=10, check_same_thread=False)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()

    
# ================================
# ROUTES
# ================================

# ================================
# ROUTES
# ================================

# # Helper function to detect language
# def get_language():
#     lang = request.args.get("lang", "en")
#     if lang not in ["en", "hi", "mr"]:
#         lang = "en"
#     return lang


@app.route("/")
def home():
    lang = get_language()
    return render_template("home.html", lang=lang)


@app.route("/predict")
def predict_page():

    if "user" not in session:
        return redirect(url_for("login"))

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

#  For Login
@app.route("/uploading_images/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db", timeout=10, check_same_thread=False)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cursor.fetchone()

        conn.close()

        if user and check_password_hash(user[3], password):

            # 🔐 Block unverified users
            if user[7] == 0:
                return render_template(
                    "login.html",
                    error="Please verify your email before logging in."
                )

            session["user"] = user[1]
            session["role"] = user[4]
            return redirect(url_for("home"))

        else:
            return render_template("login.html", error="Invalid Username or Password")

    return render_template("login.html")
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        role = request.form["role"]

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO users (username, email, password, role) VALUES (?, ?, ?, ?)",
                (username, email, hashed_password, role)
            )

            conn.commit()
            conn.close()

            # ================= EMAIL VERIFICATION =================

            token = serializer.dumps(email, salt='email-confirm')

            verification_link = url_for(
                'verify_email',
                token=token,
                _external=True
            )

            msg = Message(
                'Verify Your Email - Crop Health Diagnostic System',
                sender=app.config['MAIL_USERNAME'],
                recipients=[email]
            )

            msg.body = f"""
Hello {username},

Thank you for registering in Crop Health Diagnostic System.

Please click the link below to verify your email:

{verification_link}

This link will expire in 30 minutes.
"""

            mail.send(msg)

            # ======================================================

            return render_template(
                "register.html",
                success="Registration successful! Please check your email to verify your account."
            )

        except sqlite3.IntegrityError:
            return render_template(
                "register.html",
                error="Username or Email already exists"
            )

    return render_template("register.html")

#  Email verification route
@app.route('/verify/<token>')
def verify_email(token):
    try:
        # Decode the token (valid for 30 minutes = 1800 seconds)
        email = serializer.loads(token, salt='email-confirm', max_age=1800)
    except:
        return "Verification link is invalid or has expired."

    try:
        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE users SET is_verified = 1 WHERE email = ?",
            (email,)
        )

        conn.commit()
        conn.close()

        return "Email verified successfully! You can now login."

    except Exception as e:
        return "Database error occurred during verification."
    
    
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():

    if request.method == "POST":
        email = request.form["email"]

        conn = sqlite3.connect("users.db", timeout=10)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()

        conn.close()

        # Even if email not found, show same message (security reason)
        if user:
            token = serializer.dumps(email, salt='password-reset')

            reset_link = url_for(
                'reset_password',
                token=token,
                _external=True
            )

            msg = Message(
                'Reset Your Password - Crop Health Diagnostic System',
                sender=app.config['MAIL_USERNAME'],
                recipients=[email]
            )

            msg.body = f"""
Hello,

Click the link below to reset your password:

{reset_link}

This link will expire in 30 minutes.
"""

            mail.send(msg)

        return render_template(
            "forgot_password.html",
            success="If this email is registered, a reset link has been sent."
        )

    return render_template("forgot_password.html")

@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):

    try:
        email = serializer.loads(token, salt='password-reset', max_age=1800)
    except:
        return "Reset link is invalid or has expired."

    if request.method == "POST":
        new_password = request.form["password"]
        hashed_password = generate_password_hash(new_password)

        conn = sqlite3.connect("users.db", timeout=10)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE users SET password=? WHERE email=?",
            (hashed_password, email)
        )

        conn.commit()
        conn.close()

        return "Password updated successfully! You can now login."

    return render_template("reset_password.html")
# @app.route("/predict")
# def predict_page():
#     if "user" not in session:
#         return redirect(url_for("login"))
#     return render_template("predict.html")
# ================================
# RUN SERVER
# ================================
if __name__ == "__main__":
    init_db()
    app.run(debug=True, use_reloader=False)