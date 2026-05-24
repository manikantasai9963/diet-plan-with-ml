#  NutriGuide AI

## Personalized Diet & Smart Food Recommendation System

NutriGuide AI is a machine learning–powered nutrition recommendation web application that helps users make healthier food decisions according to their body condition, allergies, and fitness goals.

The application generates personalized weekly diet plans, dynamically adjusts meals, and predicts whether foods are healthy or unhealthy using a trained Random Forest Machine Learning model.

---

#  Features

* ✅ Personalized weekly diet planner
* ✅ BMI & calorie calculation
* ✅ Protein, fat, sugar & sodium analysis
* ✅ Allergy & dietary restriction filtering
* ✅ Dynamic meal add/remove system
* ✅ “Can I Eat This?” food checker
* ✅ Machine Learning food classification
* ✅ Interactive dashboard & nutrition charts
* ✅ Responsive modern UI

---

#  Machine Learning

The project uses a **Random Forest Classifier** trained on food nutrition datasets.

### ML Features Used

* Calories
* Fat
* Sugar
* Sodium
* Protein

### Output Classes

* Healthy
* Moderate
* Unhealthy

### Model Performance

* Accuracy: **99.56%**
* Algorithm: **Random Forest Classifier**
* Training Samples: **13,758**
* Test Samples: **3,440**

---

#  Tech Stack

## Frontend

* HTML
* CSS
* JavaScript
* Chart.js

## Backend

* Python
* Flask
* Flask-CORS

## Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib

---

#  Project Structure

```bash
NutriGuide-AI/
│
├── app.py
├── train_model.py
├── model.pkl
├── model_meta.json
├── food_db.json
├── NutriGuide_AI_ML.html
│
├── datasets/
│   ├── Indian_Food_Nutrition_Processed.csv
│   ├── cleaned_nutrition_dataset_per100g.csv
│   ├── filtered_meals.csv
│   ├── food.csv
│   └── test.csv
```

---

 Installation & Setup

 Clone Repository

```bash
git clone https://github.com/manikantasai9963/diet-plan-with-ml.git
cd diet-plan-with-ml
```

---

 Install Dependencies

```bash
pip install flask flask-cors scikit-learn pandas numpy joblib
```

---

 Run Backend

```bash
python app.py
```

Backend runs on:

```bash
http://localhost:5000
```

---

 Run Frontend

Open:

```bash
NutriGuide_AI_ML.html
```

using:

* Live Server
  OR
* browser directly

---

 Dataset Sources

The project uses multiple food nutrition datasets containing:

* calories,
* fats,
* sugar,
* sodium,
* proteins,
* Indian food nutritional values.

Datasets were merged and processed for ML training and food recommendation.



 Project Objective

The goal of NutriGuide AI is to create an interactive and intelligent nutrition assistant that:

* promotes healthier eating habits,
* provides personalized diet suggestions,
* and demonstrates practical machine learning integration in a real-world web application.



 Disclaimer

This application is recommendation-based and intended for educational purposes only.
It is not a medical diagnosis or professional healthcare system.

---

 Author

**M. Manikanta Sai**
B.Tech CSE (DS & ML)

---
