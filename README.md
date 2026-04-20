# Food Delivery Time Prediction

A machine learning project that predicts food delivery time based on real-world order data.

Course: CS 451 — Introduction to Data Science, University of Alabama, Spring 2026

---

## Problem

Food delivery platforms like Swiggy and Zomato show customers an estimated arrival time the moment an order is placed. This project replicates that system by building a regression model trained on ~45,000 real delivery records from India.

Target variable: actual delivery time in minutes (range: 10–54 min)

---

## Dataset

Source: Kaggle — Food Delivery Dataset
Link: https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset
Size: ~45,000 rows, 19 columns
License: Public (Kaggle open dataset). Fully anonymised — no PII present.

---

## Best Model Results

Model: Random Forest Regressor (tuned)
R² = 0.834
MAE = 3.07 minutes
RMSE = 3.82 minutes

---

## How to Run the Notebook

1. Upload Project_demo.ipynb to Google Colab
2. Mount your Google Drive when prompted
3. Place train.csv in: MyDrive/Food_Delivery_Dataset1/train.csv
   (Download train.csv from the Kaggle link above)
4. Run all cells top to bottom — Runtime > Run all
5. The trained model saves automatically to your Drive as best_rf_model.pkl

---

## How to Run the Gradio App

1. Make sure best_rf_model.pkl exists (run the notebook first)
2. Install dependencies: pip install -r requirements.txt
3. Run: python app.py
4. Open your browser at: http://localhost:7860

---

## Pipeline

Step 1 - Data Loading: Load raw CSV from Kaggle
Step 2 - Data Audit: Check nulls, data types, string NaNs
Step 3 - Data Cleaning: Fix target format, strip prefixes, drop bad GPS rows, impute nulls
Step 4 - Feature Engineering: Create 17 features — distance, prep time, rush hour, encodings
Step 5 - EDA: 8 visualisation plots exploring all key relationships
Step 6 - Modelling: Train/test split, Random Forest, Linear Regression baseline
Step 7 - Tuning: RandomizedSearchCV, 20 iterations, 3-fold cross-validation
Step 8 - Deployment: Gradio app for live predictions

---

## Model Comparison

Linear Regression (baseline): MAE ~4.8, RMSE ~6.1, R² ~0.61
Random Forest (default):      MAE 3.14, RMSE 3.93, R² 0.824
Random Forest (tuned):        MAE 3.07, RMSE 3.82, R² 0.834

---

## Team

[Your Name] — Data collection, cleaning, feature engineering, EDA, presentation slides 1-4
[Teammate Name] — Modelling, tuning, evaluation, Gradio deployment, presentation slides 5-8

---

## Requirements

See requirements.txt for full list.
Main libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, gradio

---

## AI Usage

AI tools (Claude by Anthropic) were used for guidance on cleaning strategies, feature engineering, tuning code, and presentation preparation. All usage is documented in Appendix A of the final report per course policy.
