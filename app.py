from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app=Flask(__name__)
model=joblib.load("models/customer_churn_model.pkl")

required_fields = [
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges"
]

def make_prediction(customer_df):
    probability = model.predict_proba(customer_df)[0, 1]

    prediction = "Yes" if probability >= 0.4 else "No"

    risk = "At Risk" if prediction == "Yes" else "Lower Risk"

    return probability, prediction, risk

def validate_customer_data(customer_data):

    if customer_data["Tenure Months"] < 0:
        return "Tenure Months cannot be negative."

    if customer_data["Monthly Charges"] < 0:
        return "Monthly Charges cannot be negative."

    if customer_data["Total Charges"] < 0:
        return "Total Charges cannot be negative."

    return None

@app.route("/health")
def health():
    return {"status": "healthy"}

@app.route("/", methods=["GET", "POST"])
def home():

    if request.method == "POST":

        customer_data = {
            "Gender": request.form["Gender"],
            "Senior Citizen": request.form["Senior Citizen"],
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "Tenure Months": float(request.form["Tenure Months"]),
            "Phone Service": request.form["Phone Service"],
            "Multiple Lines": request.form["Multiple Lines"],
            "Internet Service": request.form["Internet Service"],
            "Online Security": request.form["Online Security"],
            "Online Backup": request.form["Online Backup"],
            "Device Protection": request.form["Device Protection"],
            "Tech Support": request.form["Tech Support"],
            "Streaming TV": request.form["Streaming TV"],
            "Streaming Movies": request.form["Streaming Movies"],
            "Contract": request.form["Contract"],
            "Paperless Billing": request.form["Paperless Billing"],
            "Payment Method": request.form["Payment Method"],
            "Monthly Charges": float(request.form["Monthly Charges"]),
            "Total Charges": float(request.form["Total Charges"])
        }

        customer_df = pd.DataFrame([customer_data])

        error = validate_customer_data(customer_data)

        if error:
            return render_template("index.html", error=error)

        probability, prediction, risk = make_prediction(customer_df)

        return render_template(
            "index.html",
            probability=round(probability * 100, 2),
            prediction=prediction,
            risk=risk
        )
    
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if data is None:
        return jsonify({
            "error": "Request must contain valid JSON."
        }), 400

    missing_fields = [
        field for field in required_fields
        if field not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields.",
            "fields": missing_fields
        }), 400

    try:
        data["Tenure Months"] = float(data["Tenure Months"])
        data["Monthly Charges"] = float(data["Monthly Charges"])
        data["Total Charges"] = float(data["Total Charges"])
    except (ValueError, TypeError):
        return jsonify({
            "error": "Tenure Months, Monthly Charges, and Total Charges must be numbers."
        }), 400

    error = validate_customer_data(data)

    if error:
        return jsonify({
            "error": error
        }), 400

    customer_df = pd.DataFrame([data])

    probability, prediction, risk = make_prediction(customer_df)

    return jsonify({
        "churn_probability": round(probability * 100, 2),
        "prediction": prediction,
        "risk": risk
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)