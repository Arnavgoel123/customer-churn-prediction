# Customer Churn Prediction

An end-to-end machine learning project that predicts whether a telecom customer is likely to churn.

## Overview

Customer churn is an important business problem for telecom companies. Identifying customers who are likely to leave can help businesses take proactive retention measures.

This project builds a supervised machine learning classification system to predict customer churn using customer demographics, service information, contract details, and billing information.

## Dataset

The dataset contains information about 7,043 telecom customers, including:

- Customer demographics
- Tenure
- Phone and internet services
- Online services
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn status

During data cleaning, 11 records with missing `Total Charges` values were removed, leaving **7,032 customer records** for modeling.

## Project Workflow

1. Data loading
2. Data understanding and cleaning
3. Exploratory data analysis
4. Feature selection
5. Train-test split
6. Feature preprocessing
7. Model training
8. Model comparison
9. Hyperparameter tuning
10. Threshold analysis
11. ROC-AUC evaluation
12. Feature interpretation
13. Model saving and prediction

## Exploratory Data Analysis

Exploratory analysis revealed several patterns in customer churn:

- Customers who churned had a lower average tenure than customers who remained.
- Churned customers had higher average monthly charges.
- Month-to-month contracts had substantially higher churn rates than one-year and two-year contracts.
- Customers using fiber optic internet had a higher churn rate than customers using DSL or no internet service.

These observations represent associations in the dataset and do not imply causation.

## Data Preprocessing

The following preprocessing techniques were used:

- Numerical features were standardized using `StandardScaler`.
- Categorical features were converted using `OneHotEncoder`.
- A `ColumnTransformer` was used to apply the appropriate transformation to each feature type.
- Preprocessing and modeling were combined using a Scikit-learn `Pipeline`.
- The dataset was divided into training and testing sets using an 80/20 split.
- Stratification was used to preserve the churn class distribution.

## Models

Three classification algorithms were evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting

### Baseline Results

| Model | Accuracy | Churn F1 |
|---|---:|---:|
| Logistic Regression | 80.45% | 62% |
| Random Forest | 79.03% | 56% |
| Gradient Boosting | 79.89% | 59% |

Logistic Regression achieved the strongest baseline performance and was selected for further tuning.

## Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was used to tune Logistic Regression and Random Forest.

The models were optimized using the F1-score of the churn class (`Yes`) rather than accuracy alone.

The best Logistic Regression hyperparameter was:

```text
C = 10
```

The tuned Logistic Regression model achieved approximately:

- **Accuracy:** 80.45%
- **Churn Precision:** 64%
- **Churn Recall:** 60%
- **Churn F1-score:** 62%
- **ROC-AUC:** 0.843

## Threshold Analysis

The default classification threshold for Logistic Regression is 0.5.

Since identifying potential churners is important for customer retention, different probability thresholds were evaluated.

At the default threshold of **0.5**, churn recall was approximately **60%**.

Lowering the threshold to **0.4** increased churn recall to approximately **70%**, allowing the model to identify more customers who may be at risk of leaving.

This improvement comes at the cost of lower precision, meaning more non-churning customers may also be classified as at risk.

The appropriate threshold therefore depends on the business cost of missing a churner compared with the cost of unnecessary retention efforts.

## ROC-AUC

The final Logistic Regression model achieved a:

**ROC-AUC score of 0.843**

This indicates that the model has good ability to distinguish between customers who churn and customers who remain across different classification thresholds.

## Feature Interpretation

Logistic Regression coefficients were examined to understand which features were associated with higher or lower predicted churn probability.

Some features with stronger positive associations with churn included:

- Fiber optic internet service
- Streaming TV
- Streaming Movies
- Multiple lines
- Electronic check payment

Features with stronger negative associations included:

- Longer customer tenure
- Two-year contracts
- Having dependents
- One-year contracts

Model coefficients represent associations while controlling for other features in the model and should not be interpreted as proof of causation.

## Final Model

The final model is a tuned **Logistic Regression** classifier combined with the complete preprocessing pipeline.

For a customer-retention use case, a probability threshold of **0.4** is used to identify customers who may be at risk of churn.

The complete preprocessing and prediction pipeline was saved using Joblib so that new predictions can be generated without retraining the model.

## Example Prediction

The trained model was tested using a new customer record.

Example output:

```text
Churn Probability: 80.7%
Customer Risk: At Risk
```

This demonstrates how the saved model can be used to estimate churn risk for new customers.

## Project Structure

```text
customer-churn-prediction/
│
├── models/
│   └── customer_churn_model.pkl
│
├── notebooks/
│   └── customer_churn.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook / Google Colab
- Joblib
- Git & GitHub

## Future Improvements

Possible improvements to the project include:

- Deploying the trained model as a web application
- Experimenting with additional classification algorithms
- Performing additional feature engineering
- Selecting the classification threshold using explicit business costs
- Adding model monitoring for a deployed system

## Key Takeaway

The project demonstrates an end-to-end machine learning workflow, from raw customer data and exploratory analysis through preprocessing, model comparison, hyperparameter tuning, evaluation, interpretation, and reusable churn prediction.
