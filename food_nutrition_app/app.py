import os
import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_PATH = r"E:\Smart_Nutrition_Detection\runs\detect\lr_forced_adamw2\weights\best.pt"  # update if needed
FOOD_WEIGHT_CSV = os.path.join(BASE_DIR, "food_weight.csv")
FOOD_NUTRITION_CSV = os.path.join(BASE_DIR, "food_nutrition.csv")
ALLOWED_EXT = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = "replace_this_with_a_random_secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

# -----------------------------
# DB helpers
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            age INTEGER,
            height REAL,
            weight REAL,
            goal TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def get_db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# -----------------------------
# Load model + CSVs
# -----------------------------
# Load YOLO model lazily to speed startup if model missing
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
else:
    model = None

# Load dataframes
if os.path.exists(FOOD_WEIGHT_CSV) and os.path.exists(FOOD_NUTRITION_CSV):
    weight_df = pd.read_csv(FOOD_WEIGHT_CSV)    # columns: food_class, weight_g
    nutrition_df = pd.read_csv(FOOD_NUTRITION_CSV)  # columns: food_class, calories, protein, fat, carbs, iron, fiber, zinc
    data_df = pd.merge(weight_df, nutrition_df, on="food_class", how="inner")
else:
    data_df = pd.DataFrame()  # empty; app will still run but estimations will return zeros

# -----------------------------
# Helper functions
# -----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def estimate_weight(food_class):
    if data_df.empty:
        return 0.0
    row = data_df[data_df["food_class"] == food_class]
    return float(row.iloc[0]["weight_g"]) if not row.empty else 0.0

def estimate_nutrition(weight_g, food_class):
    if data_df.empty:
        return {k:0.0 for k in ["calories","protein","fat","carbs","iron","fiber","zinc"]}
    row = data_df[data_df["food_class"] == food_class]
    if row.empty:
        return {k:0.0 for k in ["calories","protein","fat","carbs","iron","fiber","zinc"]}
    factor = weight_g / 100.0
    return {
        "calories": float(row.iloc[0].get("calories",0)) * factor,
        "protein": float(row.iloc[0].get("protein",0)) * factor,
        "fat": float(row.iloc[0].get("fat",0)) * factor,
        "carbs": float(row.iloc[0].get("carbs",0)) * factor,
        "iron": float(row.iloc[0].get("iron",0)) * factor,
        "fiber": float(row.iloc[0].get("fiber",0)) * factor,
        "zinc": float(row.iloc[0].get("zinc",0)) * factor
    }

def process_image_return_summary(img_path):
    """Run YOLO detection, save annotated image, and return summary data"""
    if model is None:
        return [], None

    # Run detection
    results = model(img_path)
    det = results[0]

    # Save annotated image with bounding boxes
    annotated_path = os.path.join(app.config["UPLOAD_FOLDER"], "detected_" + os.path.basename(img_path))
    det.save(filename=annotated_path)

    # Handle no detections
    if not hasattr(det, "boxes") or len(det.boxes) == 0:
        return [], annotated_path

    classes = det.boxes.cls.cpu().numpy()
    summary = []
    for cls_id in classes:
        food_class = model.names[int(cls_id)]
        weight_g = estimate_weight(food_class)
        nutrition = estimate_nutrition(weight_g, food_class)
        row = {
            "food_class": food_class,
            "weight_g": round(weight_g, 2),
            "calories": round(nutrition["calories"], 2),
            "protein": round(nutrition["protein"], 2),
            "fat": round(nutrition["fat"], 2),
            "carbs": round(nutrition["carbs"], 2),
            "iron": round(nutrition["iron"], 2),
            "fiber": round(nutrition["fiber"], 2),
            "zinc": round(nutrition["zinc"], 2)
        }
        summary.append(row)

    return summary, annotated_path

# -----------------------------
# ROUTES
# -----------------------------
@app.route("/")
def index():
    return redirect(url_for("login"))

# SIGNUP (username + password only)
@app.route("/signup", methods=["GET","POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","").strip()
        if not username or not password:
            return render_template("signup.html", error="Both fields required.")
        hashed = generate_password_hash(password)
        conn = get_db_conn()
        try:
            conn.execute("INSERT INTO users (username,password) VALUES (?,?)", (username, hashed))
            conn.commit()
            conn.close()
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            conn.close()
            return render_template("signup.html", error="Username already exists.")
    return render_template("signup.html")

# LOGIN (username + password only)
@app.route("/login", methods=["GET","POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username","").strip()
        password = request.form.get("password","").strip()
        conn = get_db_conn()
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        conn.close()
        if row and check_password_hash(row["password"], password):
            session.clear()
            session["user_id"] = row["id"]
            session["username"] = row["username"]
            return redirect(url_for("choice"))
        return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")

# choice page after login: View Dashboard or Track Nutrition
@app.route("/choice")
def choice():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("choice.html")

# Dashboard: view & update age,height,weight,goal
@app.route("/dashboard", methods=["GET","POST"])
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    uid = session["user_id"]
    conn = get_db_conn()
    if request.method == "POST":
        age = request.form.get("age") or None
        height = request.form.get("height") or None
        weight = request.form.get("weight") or None
        goal = request.form.get("goal") or None
        conn.execute("UPDATE users SET age=?, height=?, weight=?, goal=? WHERE id=?", (age, height, weight, goal, uid))
        conn.commit()
    user = conn.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    conn.close()
    return render_template("dashboard.html", user=user)

# Track: upload image page
@app.route("/track", methods=["GET","POST"])
def track():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if request.method == "POST":
        if "food_image" not in request.files:
            return render_template("track.html", error="No file part.")
        file = request.files["food_image"]
        if file.filename == "" or not allowed_file(file.filename):
            return render_template("track.html", error="Invalid file type.")
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        # show loading page while processing: redirect to processing route
        session["last_uploaded"] = filename
        return redirect(url_for("processing"))
    return render_template("track.html")

# Processing page route (shows loader then performs estimation)
@app.route("/processing")
def processing():
    if "last_uploaded" not in session:
        return redirect(url_for("track"))
    # Render a page that shows loading/estimating emoji and auto-redirects to results route
    return render_template("processing.html")

# Results route performs inference and shows results
@app.route("/results")
def results():
    if "last_uploaded" not in session:
        return redirect(url_for("track"))

    filename = session["last_uploaded"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    summary, annotated_path = process_image_return_summary(file_path)

    totals = {
        "weight_g": round(sum([row["weight_g"] for row in summary]), 2),
        "calories": round(sum([row["calories"] for row in summary]), 2),
        "protein": round(sum([row["protein"] for row in summary]), 2),
        "fat": round(sum([row["fat"] for row in summary]), 2),
        "carbs": round(sum([row["carbs"] for row in summary]), 2),
        "iron": round(sum([row["iron"] for row in summary]), 2),
        "fiber": round(sum([row["fiber"] for row in summary]), 2),
        "zinc": round(sum([row["zinc"] for row in summary]), 2),
    }

    annotated_filename = os.path.basename(annotated_path) if annotated_path else None

    return render_template(
        "results.html",
        image_url=url_for('uploaded_file', filename=filename),
        detected_url=url_for('uploaded_file', filename=annotated_filename),
        summary=summary,
        totals=totals
    )

# Serve uploaded images
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Logout
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
