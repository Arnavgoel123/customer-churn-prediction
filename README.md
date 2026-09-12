# Customer Churn Prediction

An end-to-end machine learning application that predicts whether a telecom customer is likely to churn.

The project covers data analysis and model development, a Flask web application and REST API, automated testing, Docker containerization, and CI/CD with GitHub Actions.

## Overview

Customer churn is an important business problem for telecom companies. This project uses customer demographics, services, contract information, and billing data to predict churn and identify customers who may be at risk.

## Machine Learning

The dataset contains **7,043 customer records**. After removing 11 records with missing `Total Charges`, **7,032 records** were used for modeling.

The ML workflow includes:

- Data cleaning and exploratory data analysis
- Feature selection
- 80/20 stratified train-test split
- Numerical scaling with `StandardScaler`
- Categorical encoding with `OneHotEncoder`
- `ColumnTransformer` and Scikit-learn `Pipeline`
- Model comparison and hyperparameter tuning
- Probability threshold analysis
- ROC-AUC and feature interpretation

### Models

Three classification models were evaluated:

| Model | Accuracy | Churn F1 |
|---|---:|---:|
| Logistic Regression | 80.45% | 62% |
| Random Forest | 79.03% | 56% |
| Gradient Boosting | 79.89% | 59% |

Logistic Regression achieved the strongest baseline performance and was selected as the final model.

`GridSearchCV` with 5-fold cross-validation was used for tuning. The best Logistic Regression configuration used:

```text
C = 10
```

The final model achieved:

- **Accuracy:** 80.45%
- **Churn Precision:** 64%
- **Churn Recall:** 60%
- **Churn F1:** 62%
- **ROC-AUC:** 0.843

### Prediction Threshold

The default classification threshold of 0.5 was compared with lower thresholds.

A threshold of **0.4** was selected for the retention use case because it increases churn recall from approximately **60% to 70%**, allowing more potential churners to be identified at the cost of lower precision.

The complete preprocessing and model pipeline is saved using Joblib.

## Web Application

The trained model is integrated into a Flask web application where users can enter customer information and receive:

- Churn probability
- Churn prediction
- Risk level

Customers are classified as either:

- `At Risk`
- `Lower Risk`

based on the 0.4 probability threshold.

## REST API

The application exposes a REST API for programmatic predictions.

### `POST /predict`

Accepts customer information as JSON and returns the predicted churn probability, prediction, and risk level.

Example response:

```json
{
  "churn_probability": 85.94,
  "prediction": "Yes",
  "risk": "At Risk"
}
```

### `GET /health`

Returns the application health status:

```json
{
  "status": "healthy"
}
```

The API also validates required fields and numeric inputs.

## Testing

The API is tested using `pytest`.

The test suite covers:

- Health-check endpoint
- Valid prediction requests
- Missing required fields
- Invalid numeric input

Run tests with:

```bash
pytest
```

## Docker

The application is containerized using Docker.

Build the image:

```bash
docker build -t customer-churn-api .
```

Run the container:

```bash
docker run -p 5000:5000 customer-churn-api
```

The application is then available at:

```text
http://localhost:5000
```

## CI/CD

GitHub Actions automatically tests and packages the application.

```text
Git Push / Pull Request
          ↓
   Install Dependencies
          ↓
       Run pytest
          ↓
    Build Docker Image
          ↓
       Push to GHCR
```

Docker image publishing occurs only after successful tests on pushes to the `main` branch.

The image is published to GitHub Container Registry:

```text
ghcr.io/arnavgoel123/customer-churn-prediction:latest
```

## Project Structure

```text
customer-churn-prediction/
│
├── .github/
│   └── workflows/
│       └── tests.yml
├── models/
│   └── customer_churn_model.pkl
├── notebooks/
│   └── customer_churn.ipynb
├── templates/
│   └── index.html
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── app.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Technologies

- **Machine Learning:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib
- **Backend:** Flask, REST API
- **Testing:** Pytest
- **DevOps:** Docker, GitHub Actions, GitHub Container Registry
- **Development:** Jupyter Notebook / Google Colab, Git & GitHub

## Future Improvements

- Deploy the Dockerized application to a cloud platform
- Add model monitoring and periodic retraining
- Add authentication and rate limiting to the API
- Improve feature engineering and model performance
- Incorporate business costs into threshold selection

## Key Takeaway

This project demonstrates an end-to-end **ML + software engineering + DevOps workflow**, from data analysis and model development to a production-style Flask API, automated testing, Docker containerization, and CI/CD with GitHub Actions.