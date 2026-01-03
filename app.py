import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# ==============================
# LOAD DATASETS
# ==============================

# Base paths (datasets are in the project root)
BASE_DIR = Path(__file__).resolve().parent
CARDIO_PATH = BASE_DIR / "cardio_train.csv"
HEART_PATH = BASE_DIR / "heart.csv"

# Dataset 1: Cardiovascular Disease
df1 = pd.read_csv(CARDIO_PATH, sep=';')

df1 = df1[[
    'age','gender','ap_hi','cholesterol',
    'gluc','cardio'
]].copy()

df1['age'] = df1['age'] // 365
df1.columns = ['age','sex','bp','chol','sugar','target']

# Add missing features
df1['ecg'] = 0
df1['heartrate'] = 0
df1['exercise'] = 0
df1['smoking'] = 0
df1['alcohol'] = 0


# Dataset 2: Heart Disease (UCI-style heart.csv)
df2 = pd.read_csv(HEART_PATH)

# Keep compatible columns
df2 = df2[[
    'age','sex','trestbps','chol',
    'fbs','restecg','thalach',
    'exang','target'
]].copy()

df2.columns = [
    'age','sex','bp','chol',
    'sugar','ecg','heartrate',
    'exercise','target'
]

df2['smoking'] = 0
df2['alcohol'] = 0


# ==============================
# COMBINE DATASETS
# ==============================

df = pd.concat([df1, df2], ignore_index=True)

# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True)

# ==============================
# FEATURES & TARGET
# ==============================

FEATURES = [
    'age','sex','bp','chol','sugar',
    'ecg','heartrate','exercise',
    'smoking','alcohol'
]

X = df[FEATURES]
y = df['target']

# ==============================
# TRAIN / TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# MODEL TRAINING
# ==============================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

# ==============================
# SAVE MODEL
# ==============================

joblib.dump(model, 'heart_rf_model.pkl')
print("Model saved as heart_rf_model.pkl")

# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_heart_disease(
    age, sex, bp, chol, sugar,
    ecg, heartrate, exercise,
    smoking=0, alcohol=0
):
    data = np.array([[
        age, sex, bp, chol, sugar,
        ecg, heartrate, exercise,
        smoking, alcohol
    ]])

    prob = model.predict_proba(data)[0][1]
    return round(prob * 100, 2)


# ==============================
# SAMPLE PREDICTION
# ==============================

if __name__ == "__main__":
    result = predict_heart_disease(
        age=55,
        sex=1,
        bp=140,
        chol=230,
        sugar=1,
        ecg=1,
        heartrate=150,
        exercise=1,
        smoking=1,
        alcohol=0
    )

    print(f"Heart Disease Probability: {result}%")
