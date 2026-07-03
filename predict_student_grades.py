import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# We create a synthetic dataset mimicking a real school environment
np.random.seed(42)
num_students = 1000

data = {
    'study_hours_per_week': np.random.uniform(5, 30, num_students),
    'attendance_rate': np.random.uniform(60, 100, num_students),
    'sleep_hours_per_night': np.random.uniform(5, 9, num_students),
    
    'extracurricular_activities': np.random.choice(['Yes', 'No'], num_students),
    'internet_access': np.random.choice(['Yes', 'No'], num_students, p=[0.85, 0.15])
}

df = pd.DataFrame(data)

noise = np.random.normal(0, 3, num_students)
df['final_grade'] = (
    (df['study_hours_per_week'] * 1.2) + 
    (df['attendance_rate'] * 0.4) + 
    (df['sleep_hours_per_night'] * 0.5) + 
    (df['internet_access'].map({'Yes': 2, 'No': 0})) + 
    noise
)
df['final_grade'] = df['final_grade'].clip(0, 100)

print("--- Dataset Sample ---")
print(df.head(), "\n")

# SEPARATE FEATURES & TARGET
X = df.drop(columns=['final_grade'])
y = df['final_grade']

# TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PREPROCESSING PIPELINE
num_features = ['study_hours_per_week', 'attendance_rate', 'sleep_hours_per_night']
cat_features = ['extracurricular_activities', 'internet_access']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first'), cat_features)
    ]
)

# MODEL TRAINING
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = LinearRegression()
model.fit(X_train_processed, y_train)

# EVALUATION
y_pred = model.predict(X_test_processed)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("--- Model Performance Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} marks")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} marks")
print(f"R-squared Score (R²): {r2:.4f}")
print("---------------------------------")

comparison_df = pd.DataFrame({'Actual Grade': y_test.values, 'Predicted Grade': y_pred}).head()
print("\n--- Actual vs Predicted Preview ---")
print(comparison_df)