from app import app


customer_data = {
    "Gender": "Male",
    "Senior Citizen": "No",
    "Partner": "Yes",
    "Dependents": "No",
    "Tenure Months": 12,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "Yes",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 80,
    "Total Charges": 960
}


def test_health():
    client = app.test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json["status"] == "healthy"


def test_predict_valid():
    client = app.test_client()

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 200
    assert "churn_probability" in response.json
    assert "prediction" in response.json
    assert "risk" in response.json

def test_predict_missing_field():
    client = app.test_client()

    invalid_data = customer_data.copy()
    del invalid_data["Gender"]

    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 400
    assert response.json["error"] == "Missing required fields."


def test_predict_invalid_number():
    client = app.test_client()

    invalid_data = customer_data.copy()
    invalid_data["Total Charges"] = "abc"

    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 400
    assert "must be numbers" in response.json["error"]

