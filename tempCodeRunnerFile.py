import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np

data = {
    "day": [1, 2, 3, 4, 5, 6, 7],
    "temp": [31, 32, 30, 29, 30, 31, 30],
    "humidity": [80, 78, 83, 88, 85, 80, 82],
    "wind": [10, 11, 9, 14, 8, 12, 10],
    "rain_mm": [1.2, 0.0, 0.0, 6.5, 4.0, 2.0, 5.0]
}

df = pd.DataFrame(data)

print("Last 2 Days' Temperature:")
print(df.tail(2)[["day", "temp"]])

X_temp = df[["day", "humidity", "wind", "rain_mm"]]
y_temp = df["temp"]

temp_model = LinearRegression()
temp_model.fit(X_temp, y_temp)

tomorrow_features = pd.DataFrame([[8, 81, 11, 3.0]],
                                  columns=["day", "humidity", "wind", "rain_mm"])
predicted_temp = temp_model.predict(tomorrow_features)[0]

temp_lower = predicted_temp - 1
temp_upper = predicted_temp + 1
print(f"\nPredicted Temperature for Tomorrow: {predicted_temp:.2f}°C (Range: {temp_lower:.2f}°C - {temp_upper:.2f}°C)")

df["rain_today"] = df["rain_mm"].apply(lambda x: 1 if x > 0 else 0)

X_rain = df[["temp", "humidity", "wind"]]
y_rain = df["rain_today"]

rain_model = RandomForestClassifier(n_estimators=100, random_state=42)
rain_model.fit(X_rain, y_rain)

rain_features = pd.DataFrame([[predicted_temp, 81, 11]],
                              columns=["temp", "humidity", "wind"])
rain_prediction = rain_model.predict(rain_features)[0]
rain_proba = rain_model.predict_proba(rain_features)[0][1] * 100
print(f"Rain Prediction for Tomorrow: {'Yes' if rain_prediction == 1 else 'No'} ({rain_proba:.2f}% chance)")

print("\n===Predict Rain for Your Input ===")
user_temp = float(input("Enter temperature (°C): "))
user_humidity = float(input("Enter humidity (%): "))
user_wind = float(input("Enter wind speed (km/h): "))

user_input = pd.DataFrame([[user_temp, user_humidity, user_wind]],
                          columns=["temp", "humidity", "wind"])
user_rain_prediction = rain_model.predict(user_input)[0]
user_rain_proba = rain_model.predict_proba(user_input)[0][1] * 100

print(f"Rain Prediction: {'Yes' if user_rain_prediction == 1 else 'No'} ({user_rain_proba:.2f}% chance)")
