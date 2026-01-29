from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)

# =========================
# Load model and scaler
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "ridge_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# =========================
# Encode Difficulty
# =========================
def encode_difficulty(diff_value):
    diff_value = float(diff_value)

    if diff_value >= 1 and diff_value <= 2:
        return 0   # Beginner
    elif diff_value == 3:
        return 1   # Intermediate
    elif diff_value >= 4 and diff_value <= 5:
        return 2   # Advanced

# =========================
# Generate 6 AI feedbacks
# =========================
def generate_feedback(data, prediction):
    learners, rating, difficulty_raw, duration = data

    feedback = []

    if rating < 4:
        feedback.append("Improve course content quality to increase learner ratings.")
    else:
        feedback.append("Your course rating is strong. Maintain teaching quality.")

    if learners < 50000:
        feedback.append("Increase marketing and promotion to attract more learners.")
    else:
        feedback.append("Good learner engagement. Keep promoting your course.")

    if duration > 50:
        feedback.append("Consider splitting the course into smaller modules.")
    else:
        feedback.append("Course duration is well balanced.")

    if float(difficulty_raw) <= 2:
        feedback.append("Add more structured guidance and beginner-friendly examples.")
    elif float(difficulty_raw) == 3:
        feedback.append("Difficulty level is balanced. Add practical projects.")
    else:
        feedback.append("Consider adding prerequisite resources for advanced learners.")

    feedback.append("Include real-world projects and case studies.")
    feedback.append("Optimize course title and description for better search visibility.")

    return feedback

# =========================
# Routes
# =========================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        learners = float(request.form["learners"])
        rating = float(request.form["rating"])
        duration = float(request.form["duration"])
        difficulty_raw = request.form["difficulty"]

        difficulty_encoded = encode_difficulty(difficulty_raw)

        # Prepare features in correct order
        features = np.array([[learners, rating, difficulty_encoded, duration]])
        features_scaled = scaler.transform(features)

        # Predict
        prediction = round(model.predict(features_scaled)[0], 2)

        feedback = generate_feedback(
            [learners, rating, difficulty_raw, duration],
            prediction
        )

        return render_template(
            "result.html",
            prediction=prediction,
            feedback=feedback
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
