import os
import glob
import cv2
import pandas as pd
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "E:\\Smart_Nutrition_Detection\\runs\\detect\\lr_forced_adamw2\\weights\\best.pt"  # path to your trained YOLO model
IMAGE_FOLDER = r"E:\Smart_Nutrition_Detection\runs\detect\predict12"
FOOD_DENSITY_CSV = "food_density.csv"     # grams/cm2
FOOD_NUTRITION_CSV = "food_nutrition.csv" # per 100g
OUTPUT_FOLDER = "E:\\Smart_Nutrition_Detection\\runs\\detect\\visualized"  # folder to save visualized results

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# LOAD MODEL AND DATA
# -----------------------------
model = YOLO(MODEL_PATH)
density_df = pd.read_csv(FOOD_DENSITY_CSV)
nutrition_df = pd.read_csv(FOOD_NUTRITION_CSV)
data_df = pd.merge(density_df, nutrition_df, on="food_class", how="inner")

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def estimate_weight(area_cm2, food_class):
    row = data_df[data_df["food_class"] == food_class]
    if row.empty:
        return 0
    g_per_cm2 = row.iloc[0]["grams_per_cm2"]
    return area_cm2 * g_per_cm2

def estimate_nutrition(weight_g, food_class):
    row = data_df[data_df["food_class"] == food_class]
    if row.empty:
        return {k: 0 for k in ["calories","protein","fat","carbs","iron","fiber","zinc"]}
    factor = weight_g / 100.0
    return {
        "calories": row.iloc[0]["calories"] * factor,
        "protein": row.iloc[0]["protein"] * factor,
        "fat": row.iloc[0]["fat"] * factor,
        "carbs": row.iloc[0]["carbs"] * factor,
        "iron": row.iloc[0]["iron"] * factor,
        "fiber": row.iloc[0]["fiber"] * factor,
        "zinc": row.iloc[0]["zinc"] * factor
    }

# -----------------------------
# MAIN PROCESSING LOOP
# -----------------------------
image_paths = glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) + \
              glob.glob(os.path.join(IMAGE_FOLDER, "*.png"))

summary_rows = []

for img_path in image_paths:
    print(f"\n🔍 Processing: {os.path.basename(img_path)}")
    results = model(img_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()  # confidence scores

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # Draw detections
    for box, cls_id, conf in zip(boxes, classes, confs):
        x1, y1, x2, y2 = map(int, box)
        area_px = (x2 - x1) * (y2 - y1)
        area_cm2 = area_px / 100.0  # assuming 1cm = 10px → 100px² = 1cm²

        food_class = model.names[int(cls_id)]
        weight_g = estimate_weight(area_cm2, food_class)
        nutrition = estimate_nutrition(weight_g, food_class)

        summary_rows.append({
            "image": os.path.basename(img_path),
            "food_class": food_class,
            "confidence": round(float(conf), 3),
            "weight_g": round(weight_g, 2),
            **{k: round(v, 2) for k, v in nutrition.items()}
        })

        # Draw bounding box + label
        label = f"{food_class} ({conf:.2f})"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Show the image with detections
    cv2.imshow("Detections", img)
    cv2.waitKey(0)  # Press any key to show next image

    # Save the visualized image
    out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(img_path))
    cv2.imwrite(out_path, img)
    print(f"💾 Saved visualization: {out_path}")

cv2.destroyAllWindows()

# -----------------------------
# RESULTS
# -----------------------------
summary_df = pd.DataFrame(summary_rows)
if summary_df.empty:
    print("\n⚠️ No detections found.")
else:
    print("\n✅ Detailed results:\n", summary_df)

    totals = summary_df.groupby("food_class")[["weight_g","calories","protein","fat","carbs","iron","fiber","zinc"]].sum()
    grand_totals = summary_df[["calories","protein","fat","carbs","iron","fiber","zinc"]].sum()

    print("\n📊 Total nutrients per food class:\n", totals)
    print("\n🍽️ Total nutrients for the whole plate:\n", grand_totals)

    summary_df.to_csv("nutrition_estimates.csv", index=False)
    print("\n💾 Saved results to 'nutrition_estimates.csv'")
