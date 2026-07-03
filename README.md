An end-to-end regression pipeline designed to predict a student's final examination score out of 100 based on core behavioral inputs.

Core Architecture -
*   ML Task: Continuous Regression
*   Model Implemented: Baseline Linear Regression
*   Data Preprocessing: Standard Scaling (`StandardScaler`) for numerical variables; binary conversion (`OneHotEncoder`) for categorical variables.
  
Performance Metrics -
*   Mean Absolute Error (MAE): ~2.40 marks (Average prediction variance)
*   R-squared Score ($R^2$): ~0.92 (92% of variance explained by features)
