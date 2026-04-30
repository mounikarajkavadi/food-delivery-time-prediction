import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib

df = pd.read_csv('food1_model_ready.csv')

model_features = [
    'Distance_km', 'Prep_Time_min', 'Hour_of_Day', 'Is_Rush_Hour',
    'Is_Weekend', 'Delivery_person_Age', 'Delivery_person_Ratings',
    'Is_Multiple_Delivery', 'Vehicle_condition', 'Weather_Code',
    'Traffic_Code', 'Vehicle_Code', 'City_Code', 'Time_Day_Code',
    'Order_Type_Code', 'Is_Festival',
]

print("Columns in CSV:", list(df.columns))
print("Shape:", df.shape)

# Check all features exist
missing = [f for f in model_features if f not in df.columns]
print("Missing features:", missing)

X = df[model_features]
y = df['Time_taken_min']

print("\nTarget stats:")
print(y.describe())
print("\nDistance range:", X['Distance_km'].min(), "-", X['Distance_km'].max())
print("Traffic range:", X['Traffic_Code'].min(), "-", X['Traffic_Code'].max())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining model...")
model = RandomForestRegressor(
    n_estimators=300, max_depth=20, max_features=0.5,
    min_samples_split=10, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"R²  : {r2_score(y_test, y_pred):.4f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):.4f}")

# Quick sanity check
bad = pd.DataFrame([[15,10,10,0,0,28,4.5,0,1,5,3,2,1,0,2,1]], columns=model_features)
good = pd.DataFrame([[3,10,10,0,0,28,4.5,0,1,0,0,2,1,0,2,0]], columns=model_features)
print(f"\nBad conditions:  {model.predict(bad)[0]:.1f} min")
print(f"Good conditions: {model.predict(good)[0]:.1f} min")

joblib.dump(model, 'best_rf_model.pkl')
print("\nSaved best_rf_model.pkl")