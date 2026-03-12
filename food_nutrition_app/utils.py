import pandas as pd
from ultralytics import YOLO
import os

MODEL_PATH = r"E:\Smart_Nutrition_Detection\runs\detect\lr_forced_adamw2\weights\best.pt"
FOOD_WEIGHT_CSV = "food_weight.csv"
FOOD_NUTRITION_CSV = "food_nutrition.csv"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load CSV data
weight_df = pd.read_csv(FOOD_WEIGHT_CSV)
nutrition_df = pd.read_csv(FOOD_NUTRITION_CSV)

def get_weight(food_class):
    """Fetch food weight from CSV."""
    row = weight_df[weight_df['food_class'].str.lower() == food_class.lower()]
    if row.empty:
        print(f"⚠️ No weight found for: {food_class}")
        return 0
    return float(row.iloc[0]['weight_g'])

def get_nutrition(food_class):
    """Fetch nutrition data per 100g from CSV."""
    row = nutrition_df[nutrition_df['food_class'].str.lower() == food_class.lower()]
    if row.empty:
        print(f"⚠️ No nutrition found for: {food_class}")
        return {k: 0 for k in ["calories","protein","fat","carbs","iron","fiber","zinc"]}
    return row.iloc[0].to_dict()

def estimate_nutrition(weight_g, food_class):
    """Estimate total nutrition based on weight."""
    base = get_nutrition(food_class)
    factor = weight_g
    return {
        "calories": round(base["calories"] * factor, 2),
        "protein": round(base["protein"] * factor, 2),
        "fat": round(base["fat"] * factor, 2),
        "carbs": round(base["carbs"] * factor, 2),
        "iron": round(base["iron"] * factor, 2),
        "fiber": round(base["fiber"] * factor, 2),
        "zinc": round(base["zinc"] * factor, 2)
    }

def detect_nutrition(img_path):
    """Run YOLO detection and compute nutrition totals."""
    print(f"🔍 Detecting on image: {img_path}")
    if not os.path.exists(img_path):
        print("❌ Image not found at path:", img_path)
        return {"summary": [], "totals": {}}

    results = model(img_path, conf=0.25)
    summary = []

    if len(results[0].boxes) == 0:
        print("⚠️ No detections found.")
        return {"summary": [], "totals": {}}

    for box, cls_id in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        food_class = model.names[int(cls_id)]
        print(f"🍴 Detected: {food_class}")
        weight_g = get_weight(food_class)
        nutrition = estimate_nutrition(weight_g, food_class)
        summary.append({"food_class": food_class, "weight_g": weight_g, **nutrition})

    # Calculate totals
    totals = {}
    if summary:
        for key in ["weight_g", "calories","protein","fat","carbs","iron","fiber","zinc"]:
            totals[key] = round(sum(item.get(key, 0) for item in summary), 2)

    print("✅ Final totals:", totals)
    return {"summary": summary, "totals": totals}
