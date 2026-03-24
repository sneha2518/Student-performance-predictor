from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and encoders
model = pickle.load(open("model.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = request.form['gender']
        race = request.form['race']
        parent_edu = request.form['parent_edu']
        lunch = request.form['lunch']
        test_prep = request.form['test_prep']

        input_data = [
            encoders['gender'].transform([gender])[0],
            encoders['race/ethnicity'].transform([race])[0],
            encoders['parental level of education'].transform([parent_edu])[0],
            encoders['lunch'].transform([lunch])[0],
            encoders['test preparation course'].transform([test_prep])[0]
        ]

        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0]

        if prediction >= 70:
            result = "Good"
        elif prediction >= 40:
            result = "Average"
        else:
            result = "Poor"

        return render_template("index.html",
                               prediction_text=f"🎯 Performance: {result}",
                               score=f"📊 Score: {round(prediction, 2)}")

    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)