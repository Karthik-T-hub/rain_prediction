import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Dataset
data = {
    'temperature': [30, 25, 20, 18, 35, 28, 22],
    'humidity': [70, 80, 90, 95, 60, 75, 85],
    'wind_speed': [10, 12, 8, 6, 15, 9, 7],
    'rain': [0, 1, 1, 1, 0, 0, 1]
}

df = pd.DataFrame(data)

X = df[['temperature', 'humidity', 'wind_speed']]
y = df['rain']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl created successfully")