# task4-aiml
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("adult_income_dataset.csv")

categorical_cols = ["workclass", "education", "marital_status", "occupation"]
numerical_cols = ["age", "education_num", "hours_per_week"]

le = LabelEncoder()
df["income"] = le.fit_transform(df["income"])

df_encoded = pd.get_dummies(df, columns=categorical_cols)

scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

df_encoded.to_csv("adult_income_processed.csv", index=False)

print("Processed dataset saved successfully")

Why is scaling required?
Answer:
Scaling is required to bring all numerical features to a common range so that no feature dominates the model due to larger values.
What is normalization?
Answer:
Normalization is a feature scaling technique that rescales values between 0 and 1.

Which algorithms need scaling?
Answer:K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Logistic Regression
Linear Regression
Neural Networks
K-Means Clustering
PCA

What is feature engineering?
Answer:
Feature engineering is the process of creating, transforming, or selecting features to improve machine learning model performance.



​


