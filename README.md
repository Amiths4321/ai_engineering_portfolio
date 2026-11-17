
# Mumbai Property Price Prediction

An end-to-end machine learning project that
predicts residential property prices using
classical machine learning algorithms.

## Project Overview

This project explores real estate property data
and develops regression models to estimate
property prices.

The project includes:

- Exploratory Data Analysis
- Data cleaning and preprocessing
- Missing-value handling
- Outlier analysis
- Feature engineering and selection
- Regression model development
- Model evaluation and comparison
- Hyperparameter tuning
- Streamlit application

## Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Streamlit
- Joblib
- Jupyter Notebook

## Machine Learning Models

1. Linear Regression
2. Random Forest Regression
3. XGBoost Regression

## Project Structure

project_01_classical_ml/
|
├── app.py
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── requirements.txt
├── .gitignore
└── README.md

## Installation

Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

Navigate into the project:

cd project_01_classical_ml

Create a virtual environment:

python -m venv .venv

Activate on Windows:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Run the Application

streamlit run app.py

## Model Evaluation

Models are compared using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R2)

See reports/model_performance_report.md
for the detailed comparison.

## Limitations

- Predictions depend on the quality of the dataset.
- Historical listing prices may differ from
  actual transaction prices.
- Some missing values require imputation.
- Manual predictions use defaults for features
  not entered by the user.

## Future Improvements

- Improve location-based modelling.
- Add better feature engineering.
- Improve prediction reliability.
- Add automated model evaluation.
- Deploy the application.

## Author

AI Engineering Portfolio Project
