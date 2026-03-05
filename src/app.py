from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# ===== Ruta segura del modelo =====
basedir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(basedir, "model.pkl")

# Cargar modelo
with open(model_path, "rb") as f:
    model = pickle.load(f)

# ===== Página principal =====
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Recoger datos del formulario
            data = [
                float(request.form["Age"]),
                float(request.form["Family_Income"]),
                float(request.form["Study_Hours_per_Day"]),
                float(request.form["Attendance_Rate"]),
                float(request.form["Assignment_Delay_Days"]),
                float(request.form["Travel_Time_Minutes"]),
                float(request.form["Stress_Index"]),
                float(request.form["Gender_n"]),
                float(request.form["Internet_Access_n"]),
                float(request.form["Part_Time_Job_n"]),
                float(request.form["Scholarship_n"]),
                float(request.form["Semester_n"]),
                float(request.form["Department_n"]),
                float(request.form["Parental_Education_n"]),
            ]

            data = np.array(data).reshape(1, -1)

            # Hacer predicción
            prediction = model.predict(data)[0]

        except Exception as e:
            prediction = f"Error en los datos introducidos: {e}"

    return render_template("index.html", prediction=prediction)

# ===== Ejecutar app =====
if __name__ == "__main__":
    # debug=True para reinicio automático mientras desarrollas
    app.run(debug=True)
