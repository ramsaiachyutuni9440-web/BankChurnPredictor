# Bank Customer Churn Predictor — ANN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Keras](https://img.shields.io/badge/Keras-Built--in-red) ![License](https://img.shields.io/badge/License-MIT-green) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A deep learning project that predicts whether a bank customer is likely to churn (leave the bank), using an Artificial Neural Network (ANN) built with TensorFlow and Keras. Trained on the Kaggle Churn Modelling dataset of 10,000 customers.

Built as part of my hands-on ML portfolio while transitioning from IT production support into AI/ML engineering.

---

## Problem Statement

Customer churn is one of the most expensive problems in banking. Acquiring a new customer costs 5–7x more than retaining an existing one. This project builds a neural network that predicts which customers are at risk of leaving — giving banks the opportunity to intervene before it's too late.

---

## Demo

### Training Accuracy & Loss Curves
![Training History](training_history.png)

> Accuracy improves steadily across epochs. The gap between train and validation curves is minimal — indicating the model generalises well without overfitting.

### Single Customer Prediction
```
Input:  Age=40, Balance=60000, Credit Score=600, Active Member=Yes
Output: Churn probability: 18.4% → ✅ This customer is likely to STAY
```

---

## Project Structure

```
bank-churn-predictor/
│
├── data/
│   └── Churn_Modelling.csv          # Kaggle dataset (10,000 customers)
│
├── notebooks/
│   └── churn_analysis.ipynb         # Full EDA + model training notebook
│
├── churn_predictor.py               # Main Python script
├── churn_model.keras                # Saved trained ANN model
├── scaler.pkl                       # Saved StandardScaler (needed for predictions)
├── training_history.png             # Accuracy & loss plots
├── requirements.txt                 # Dependencies
└── README.md
```

---

## Dataset

- **Source:** [Churn Modelling Dataset](https://www.kaggle.com/datasets/shubh0799/churn-modelling) on Kaggle
- **Rows:** 10,000 customer records
- **Columns:** 14 features — credit score, geography, gender, age, tenure, balance, number of products, credit card status, active membership, estimated salary
- **Target variable:** `Exited` — 1 (churned) or 0 (stayed)
- **Class split:** ~20% churned, ~80% retained

---

## Model Architecture

```
Input Layer     →  11 features (after preprocessing)
Hidden Layer 1  →  64 neurons, ReLU activation + Dropout (0.3)
Hidden Layer 2  →  32 neurons, ReLU activation + Dropout (0.3)
Output Layer    →  1 neuron, Sigmoid activation
```

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs | 100 |
| Batch Size | 32 |
| Validation Split | 10% |

**Why these choices?**
- **ReLU** in hidden layers — avoids vanishing gradient, fast to compute, standard for deep learning
- **Sigmoid** in output — squishes prediction to 0–1 range, ideal for binary classification probability
- **Dropout** — randomly disables 30% of neurons during training to prevent overfitting
- **Adam** — adaptive learning rate optimiser, outperforms plain SGD on most tasks

---

## Approach

| Step | Description |
|------|-------------|
| 1. EDA | Explored churn rates by geography, age, balance, and activity status |
| 2. Preprocessing | Dropped irrelevant columns, label encoded Gender, one-hot encoded Geography |
| 3. Feature Scaling | StandardScaler applied after train/test split to prevent data leakage |
| 4. Train/Test Split | 80/20 split with `random_state=42` for reproducibility |
| 5. ANN Training | 2 hidden layers with Dropout, 100 epochs, batch size 32 |
| 6. Evaluation | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| 7. Visualisation | Training/validation accuracy and loss curves across epochs |

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~86% |
| Precision (Churn=1) | ~0.74 |
| Recall (Churn=1) | ~0.50 |
| F1-Score (Churn=1) | ~0.60 |

> **Note:** Recall on the churn class is intentionally a focus area — in a real banking scenario, a false negative (missing a churner) is more costly than a false positive. Future work will address this via class weighting or threshold tuning.

---

## Key Findings

- Customers aged **40–55** have a significantly higher churn rate
- **German customers** churn at almost double the rate of French and Spanish customers
- **Inactive members** with high balances are at the highest risk
- Having **more than 2 products** actually increases churn likelihood — possibly due to fee dissatisfaction
- **Gender** had a small but measurable effect — female customers churned slightly more often

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/bank-churn-predictor.git
cd bank-churn-predictor
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/shubh0799/churn-modelling) and place `Churn_Modelling.csv` in the `data/` folder.

**4. Run the script**
```bash
python churn_predictor.py
```

**5. Predict for a new customer**
```python
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model('churn_model.keras')
sc    = joblib.load('scaler.pkl')

# [CreditScore, Gender, Age, Tenure, Balance,
#  NumOfProducts, HasCrCard, IsActiveMember,
#  EstimatedSalary, Geography_Germany, Geography_Spain]
new_customer = [[600, 1, 40, 3, 60000, 2, 1, 1, 50000, 0, 1]]
prob = model.predict(sc.transform(new_customer))[0][0]
print(f"Churn probability: {prob:.2%}")
```

---

## Requirements

```
tensorflow>=2.10
pandas>=1.5
scikit-learn>=1.3
matplotlib>=3.6
seaborn>=0.12
joblib>=1.2
numpy>=1.23
```

---

## What I Learned

- Designing and training an ANN from scratch using TensorFlow and Keras
- Why feature scaling is mandatory for neural networks and how data leakage happens if done incorrectly
- The difference between label encoding and one-hot encoding, and when to use each
- How to interpret training/validation curves to diagnose overfitting and underfitting
- The trade-off between precision and recall in imbalanced classification problems
- Why accuracy alone is a misleading metric when classes are imbalanced

---

## Future Improvements

- [ ] Address class imbalance using SMOTE or class_weight parameter
- [ ] Tune decision threshold from 0.5 to optimise recall for the churn class
- [ ] Experiment with deeper architectures and compare performance
- [ ] Add hyperparameter tuning with Keras Tuner
- [ ] Build a Streamlit web app for interactive single-customer predictions
- [ ] Add SHAP values to explain individual predictions

---

## About Me

I'm an IT production support professional with 4 years of mainframe experience, actively building an AI/ML portfolio as I transition into machine learning engineering. This is my second ML project — following an employee attrition predictor built with scikit-learn.

- LinkedIn: [your-linkedin-url]
- GitHub: [your-github-url]

---

## License

This project is licensed under the MIT License. The Churn Modelling dataset is publicly available on Kaggle for educational use.
