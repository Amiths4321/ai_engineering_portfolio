
# Mumbai Property Price Prediction
## Machine Learning Performance Report

**Report date:** 2026-11-17

## 1. Project Overview

This project develops a machine learning model
to predict residential property prices using
property-related information.

**Target variable:** Price

**Dataset size:** 12,685 original records

## 2. Data Preparation

The dataset was examined for:
- Missing values
- Duplicate records
- Data types
- Price distribution
- Potential outliers
- Data leakage

The initial modelling dataset excluded records
with missing target prices.

Categorical features were encoded using
OneHotEncoder.

Numerical missing values were handled using
median imputation.

## 3. Models Evaluated

- Linear Regression
- Random Forest Regression
- XGBoost Regression
- Tuned best model

## 4. Model Performance

            Model          MAE         RMSE          R2
Linear Regression 1.603213e+07 7.303623e+07   -5.040484
    Random Forest 2.004439e+07 4.601505e+08 -238.769988
          XGBoost 2.179448e+06 8.819066e+06    0.911928
 Tuned Best Model 1.603213e+07 7.303623e+07   -5.040484

## 5. Best Model

The lowest test MAE in this comparison belongs to:

**XGBoost**

MAE: ₹2,179,447.97

RMSE: ₹8,819,066.06

R2: 0.9119

## 6. Limitations

- Missing values may affect prediction quality.
- Historical listing prices may not represent
  actual transaction prices.
- Some categorical features may have many unique values.
- Manual application inputs use representative
  defaults for unspecified features.
- Model performance may vary across locations
  and property categories.

## 7. Future Improvements

- Improve location-based features.
- Investigate outliers and data quality.
- Perform more robust cross-validation.
- Improve hyperparameter tuning.
- Add prediction uncertainty estimates.
- Deploy the application with proper validation.

## 8. Conclusion

This project demonstrates an end-to-end
classical machine learning workflow, including
data analysis, preprocessing, model comparison,
model persistence and application development.
