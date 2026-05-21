from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    temp = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    wind = float(request.form["wind_speed"])

    features = np.array([[temp, humidity, wind]])
    prediction = model.predict(features)

    result = "Rain Expected 🌧️" if prediction[0] == 1 else "No Rain ☀️"

    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)