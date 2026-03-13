# NutriX - Smart Nutrition Detection System

NutriX is an AI-powered nutrition analysis system that detects food items from plate images and estimates their nutritional composition automatically. The system uses a custom-trained YOLOv8 object detection model combined with a web interface built using Flask.

The application allows users to upload a food plate image and instantly receive detected food items along with estimated calories, protein, fat, carbohydrates, iron, fiber, and zinc.

Live Demo:  
https://dinesh012-smart-nutrition-detection.hf.space

---

## Features

• Multi-class food detection using YOLOv8  
• Automated nutrition estimation from detected food items  
• User authentication (signup/login)  
• Personalized dashboard for storing user health details  
• Bounding box visualization on detected food items  
• Real-time inference through a Flask web interface  
• Public deployment on Hugging Face Spaces  

---

## Technologies Used

**Machine Learning**
- YOLOv8 (Ultralytics)
- PyTorch

**Backend**
- Flask
- Python

**Computer Vision**
- OpenCV
- Pillow

**Data Processing**
- Pandas
- NumPy

**Deployment**
- Hugging Face Spaces
- Docker

---

## Model Training Details

- Dataset Size: 12,000 food images  
- Number of Classes: 31  
- Training Epochs: 50  
- Optimizer: AdamW  
- Learning Rate: 0.001  
- Batch Size: 8  
- Image Size: 416  
- GPU Used: NVIDIA RTX 3050  

Model Performance:
- Validation Accuracy: **91.3%**

---

## Installation

Clone the repository

```bash
git clone https://github.com/DineshKarthik12/NutriX---Computer-Vison-Based-Smart-Nutrition-Detection-System.git
cd food_nutrition_app
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows:
```bash
venv\Scripts\activate
```

Linux / Mac:
```bash
source venv/bin/activate
```

Install required dependencies

```bash
pip install -r requirements.txt
```

Ensure the trained YOLOv8 model file `best.pt` is present in the project root directory.

Run the Flask application

```bash
python app.py
```

Open the application in your browser

```
http://localhost:7860
```

---

## Usage

1. Create a new account using the signup page.
2. Log in with your credentials.
3. Upload a food plate image from the **Track Nutrition** page.
4. The system detects food items using the YOLOv8 model.
5. Detected foods are mapped to the nutrition database.
6. Nutritional values (calories, protein, fat, carbs, iron, fiber, zinc) are calculated and displayed.
7. The detected image with bounding boxes and nutrition summary is shown to the user.
