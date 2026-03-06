from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = np.array([[
        float(request.form["study_hours"]),
        float(request.form["attendance"]),
        float(request.form["delay"]),
        float(request.form["stress"]),
        float(request.form["gpa"])
    ]])

    pred = model.predict(data)[0]

    result = "🔴 Riesgo de Deserción" if pred == 1 else "🟢 Continúa Estudiando"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
    