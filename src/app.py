from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/", methods=["GET","POST"])
def home():
    
    prediction = None
    
    if request.method == "POST":
        
        age = float(request.form["age"])
        study_hours = float(request.form["study_hours"])
        attendance = float(request.form["attendance"])
        stress = float(request.form["stress"])
        gpa = float(request.form["gpa"])
        
        data = np.array([[age, study_hours, attendance, stress, gpa]])
        
        pred = model.predict(data)[0]
        
        if pred == 1:
            prediction = "Student likely to dropout"
        else:
            prediction = "Student likely to continue"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
    