import streamlit as st
import joblib
import numpy as np

# Cargar modelo
model = joblib.load("model.pkl")

st.title("🎓 Predicción de Deserción Estudiantil")

st.write("Ingrese los datos del estudiante:")

# Inputs
study_hours = st.number_input("Study Hours per Day", step=0.1)
attendance = st.number_input("Attendance Rate", step=0.1)
delay = st.number_input("Assignment Delay Days", step=0.1)
stress = st.number_input("Stress Index (1-10)", step=0.1)
gpa = st.number_input("GPA", step=0.1)

# Botón
if st.button("Predecir"):

    data = np.array([[study_hours, attendance, delay, stress, gpa]])
    
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("🔴 Riesgo de Deserción")
    else:
        st.success("🟢 Continúa Estudiando")