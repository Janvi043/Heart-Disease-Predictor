# ❤️ Heart Disease Risk Predictor

An AI-powered web application that predicts the risk of heart disease using machine learning. Built with Streamlit and Random Forest Classifier, this tool provides an intuitive interface for clinical risk assessment.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)

## ✨ Features

- **Interactive Web Interface**: User-friendly Streamlit-based UI with dark theme
- **Real-time Predictions**: Instant heart disease risk assessment
- **Comprehensive Input**: Supports multiple clinical parameters including:
  - Age, Sex, Blood Pressure
  - Cholesterol levels, Blood Sugar
  - ECG results, Heart Rate
  - Lifestyle factors (Exercise, Smoking, Alcohol)
- **Visual Risk Display**: Circular gauge visualization for risk percentage
- **High/Low Risk Classification**: Clear categorization of results

## 📖 About

This project combines two cardiovascular datasets to train a robust Random Forest Classifier model for predicting heart disease risk. The application provides healthcare professionals and individuals with a tool to assess cardiovascular risk based on clinical parameters.

## 📊 Dataset

The model is trained on a combined dataset from:
1. **Cardiovascular Disease Dataset** (`cardio_train.csv`)
2. **Heart Disease Dataset** (`heart.csv` - UCI-style)

The datasets are merged and standardized to create a comprehensive training set with the following features:
- Age (years)
- Sex (0: Female, 1: Male)
- Blood Pressure (mm Hg)
- Cholesterol (mg/dl)
- Fasting Blood Sugar
- Resting ECG results
- Maximum Heart Rate
- Exercise-induced angina
- Lifestyle factors (Smoking, Alcohol)

## 🤖 Model

- **Algorithm**: Random Forest Classifier
- **Parameters**:
  - `n_estimators`: 300
  - `max_depth`: 10
  - `random_state`: 42
- **Evaluation Metrics**:
  - Accuracy Score
  - ROC-AUC Score
- **Model File**: `heart_rf_model.pkl`

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Janvi043/Heart-Disease-Predictor.git
   cd Heart-Disease-Predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (if needed)
   ```bash
   python app.py
   ```
   This will train the model and save it as `heart_rf_model.pkl`.

## 💻 Usage

### Running the Web Application

1. **Start the Streamlit app**
   ```bash
   streamlit run front.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If it doesn't, manually navigate to the URL shown in the terminal

3. **Using the Application**
   - Fill in the patient details in the left panel
   - Adjust sliders for clinical parameters
   - Optionally expand "Lifestyle Factors" for additional inputs
   - Click "🔍 Predict Risk" to get the prediction
   - View the risk percentage and classification

### Direct Model Usage

You can also use the model directly in Python:

```python
from app import predict_heart_disease

# Example prediction
risk_percentage = predict_heart_disease(
    age=55,
    sex=1,  # 1: Male, 0: Female
    bp=140,
    chol=230,
    sugar=1,  # 1: >120 mg/dl, 0: <=120 mg/dl
    ecg=1,  # 1: Abnormal, 0: Normal
    heartrate=150,
    exercise=1,  # 1: Yes, 0: No
    smoking=1,  # 1: Yes, 0: No
    alcohol=0  # 1: Yes, 0: No
)

print(f"Heart Disease Risk: {risk_percentage}%")
```

## 📁 Project Structure

```
Heart-Disease-Predictor/
│
├── app.py                 # Model training and prediction logic
├── front.py              # Streamlit web application
├── heart_rf_model.pkl    # Trained Random Forest model
├── cardio_train.csv      # Cardiovascular dataset
├── heart.csv             # Heart disease dataset
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore file
└── README.md            # Project documentation
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
  - RandomForestClassifier
  - train_test_split
  - accuracy_score, roc_auc_score
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **joblib**: Model serialization


## 👤 Author

**Janvi043**

- GitHub: [@Janvi043](https://github.com/Janvi043)

---

**Built for Kaggle Royale • Educational Use Only**

---
*Last updated: Repository contributors updated*
