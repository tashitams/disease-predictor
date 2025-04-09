import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import numpy as np

# Load dataset
df = pd.read_csv('data/dataset.csv')

# --- Data Preprocessing ---

# 1. Convert Yes/No to 1/0 for symptom columns (updated to use map instead of applymap)
symptom_cols = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing']
for col in symptom_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0}).astype(int)

# 2. Map Blood Pressure & Cholesterol Level
bp_map = {'Low': 0, 'Normal': 1, 'High': 2}
df['Blood Pressure'] = df['Blood Pressure'].map(bp_map)
df['Cholesterol Level'] = df['Cholesterol Level'].map(bp_map)

# 3. Encode Gender
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

# 4. Encode Outcome Variable (if needed for future use)
outcome_map = {'Negative': 0, 'Positive': 1}
df['Outcome Variable'] = df['Outcome Variable'].map(outcome_map)

# 5. Check and filter rare diseases
disease_counts = df['Disease'].value_counts()
print("Disease counts before filtering:")
print(disease_counts)

# Filter out diseases with less than 2 samples
valid_diseases = disease_counts[disease_counts >= 2].index
df = df[df['Disease'].isin(valid_diseases)]

# 6. Encode Disease (Target)
le_disease = LabelEncoder()
df['Disease'] = le_disease.fit_transform(df['Disease'])
joblib.dump(le_disease, 'model/disease_encoder.pkl')

# --- Feature Selection ---
# Drop Outcome Variable if not needed for prediction
X = df.drop(['Disease', 'Outcome Variable'], axis=1)
y = df['Disease']

# --- Identify column types ---
numeric_cols = ['Age']  # Only Age is truly numeric
categorical_cols = ['Gender', 'Blood Pressure', 'Cholesterol Level']
binary_cols = symptom_cols  # Already converted to 0/1

# --- Create preprocessing pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', 'passthrough', categorical_cols),  # Already encoded
        ('binary', 'passthrough', binary_cols)     # Already 0/1
    ])

# --- Create and train model pipeline ---
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'  # Helps with imbalanced datasets
    ))
])

# --- Train-Test Split ---
# Remove stratify if classes are still too small after filtering
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # Try with stratification first
    )
except ValueError:
    print("Warning: Couldn't stratify due to small class sizes, splitting without stratification")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

# --- Train the Model ---
pipeline.fit(X_train, y_train)

# --- Save Model ---
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/disease_model.pkl')

# --- Save additional encoders/mappings ---
joblib.dump(le_gender, 'model/gender_encoder.pkl')
joblib.dump(bp_map, 'model/bp_col_encoder.pkl')
joblib.dump(outcome_map, 'model/outcome_encoder.pkl')

# --- Evaluation ---
print("\nModel training completed successfully!")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Training Accuracy: {pipeline.score(X_train, y_train):.2f}")
print(f"Test Accuracy: {pipeline.score(X_test, y_test):.2f}")
print("\nClass distribution in training set:")
print(pd.Series(y_train).value_counts())
print("\nClass distribution in test set:")
print(pd.Series(y_test).value_counts())