from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# -------------------------
# Load scaler and Ridge model
# -------------------------
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("ridge_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------------
# Function to generate AI feedback
# -------------------------
def generate_feedback(score):
    if score >= 4.5:
        return "Excellent! Your course is highly successful. Keep up the great work!"
    elif score >= 3.0:
        return ("Your course is performing moderately. Consider increasing engagement, "
                "improving ratings, or adjusting difficulty to boost success.")
    else:
        return ("Your course has a low predicted success. Focus on engagement, "
                "rating improvement, shortening duration, and making content easier to follow.")

# -------------------------
# Home page
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# Prediction route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        learners = float(request.form.get("Learners", 0))
        rating = float(request.form.get("Rating", 0))
        duration = float(request.form.get("Duration_Hours", 0))
        difficulty = float(request.form.get("Difficulty", 0))

        # Debug prints
        print("Learners:", learners, "Rating:", rating, "Duration:", duration, "Difficulty:", difficulty)

        # Create DataFrame
        df = pd.DataFrame([[learners, rating, duration, difficulty]],
                          columns=["Learners", "Rating", "Duration_Hours", "Difficulty"])

        # Scale and predict
        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)[0]

        # Generate feedback
        feedback = generate_feedback(prediction)

        return render_template("result.html",
                               prediction=round(prediction, 2),
                               feedback=feedback)
    except Exception as e:
        return f"Error: {e}"

# -------------------------
# Run Flask app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
