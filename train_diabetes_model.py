import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt



warnings.filterwarnings('ignore')


df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]


cols_with_zero_issue = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[cols_with_zero_issue] = X[cols_with_zero_issue].replace(0, np.nan)


cols_to_group_fill = ["Glucose", "Insulin"]
X[cols_to_group_fill] = X[cols_to_group_fill].fillna(df.groupby('Outcome')[cols_to_group_fill].transform('median'))

# Feature Engineering 
X['Glucose_BMI'] = X['Glucose'] * X['BMI']
X['Age_Glucose'] = X['Age'] * X['Glucose']
X['BloodPressure_BMI'] = X['BloodPressure'] * X['BMI']


X['Insulin_Glucose_Ratio'] = X['Insulin'] / (X['Glucose'] + 1e-5)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")), 
    ("scaler", RobustScaler())
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, None],
    'model__min_samples_split': [2, 5],
    'model__criterion': ['gini', 'entropy']
}

print("Training Model with Grid Search...")
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_


y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n--- Model Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))




model_file = "diabetes_best_model.pkl"
with open(model_file, "wb") as f:
    pickle.dump(best_model, f)

print(f"\n✅ Model saved as: {model_file}")