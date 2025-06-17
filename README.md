# Insurance Fraud Detection System

An automated machine learning system for detecting fraudulent insurance claims using advanced data analytics and pattern recognition.

## 🎯 Project Overview

This project aims to develop an automatic fraud detection system that:
- Analyzes insurance claims data to identify suspicious patterns
- Uses machine learning techniques to classify claims as fraudulent or legitimate
- Provides a user-friendly GUI for real-time fraud prediction
- Achieves high accuracy through advanced feature engineering and model optimization

## 📊 Features

- **Comprehensive Data Analysis**: Exploratory data analysis with robust visualizations
- **Advanced Preprocessing**: Feature engineering, scaling, and handling imbalanced data
- **Multiple ML Models**: Comparison of 10+ machine learning algorithms
- **Model Optimization**: Hyperparameter tuning and cross-validation
- **Interactive GUI**: User-friendly application for fraud prediction
- **Real-time Predictions**: Fast and accurate fraud detection
- **Export Functionality**: Save prediction results and model summaries

## 🛠️ Technology Stack

- **Python 3.7+**
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Data Analysis**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **GUI**: tkinter
- **Model Persistence**: joblib

## 🚀 Quick Start

### Prerequisites

```bash
# Install Python 3.7 or higher
# Install pip package manager
```

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the Jupyter notebook for model training**:
```bash
jupyter notebook insurance_fraud_detection.ipynb
```

4. **Execute all cells in the notebook** to:
   - Load and explore the dataset
   - Perform feature engineering
   - Train multiple ML models
   - Save the best performing model

## 📁 Project Structure

```
car insurance fraud/
├── insurance_claims.csv              # Dataset
├── insurance_fraud_detection.ipynb   # Main analysis notebook
├── requirements.txt                 # Python dependencies
├── README.md                       # Project documentation
├── fraud_detection_model.pkl       # Trained model (generated)
└── model_summary.txt               # Model performance summary (generated)
```

## 📈 Model Performance

The system compares multiple machine learning algorithms:

- **Random Forest**
- **Gradient Boosting**
- **XGBoost**
- **LightGBM**
- **Logistic Regression**
- **Support Vector Machine**
- **Naive Bayes**
- **Decision Tree**
- **K-Nearest Neighbors**
- **AdaBoost**

Performance metrics include:
- **Accuracy**: Overall prediction correctness
- **Precision**: Fraud detection precision
- **Recall**: Fraud detection coverage
- **F1-Score**: Balanced performance measure
- **ROC-AUC**: Area under the ROC curve

## 📊 Key Insights

### Fraud Indicators
The analysis identifies key fraud indicators:
- Unusually high claim amounts relative to premiums
- Specific incident patterns and collision types
- Customer demographic characteristics
- Vehicle age and incident timing patterns

### Business Impact
- **Reduced Fraudulent Payouts**: Early detection of suspicious claims
- **Faster Processing**: Automated screening of legitimate claims
- **Improved Risk Assessment**: Data-driven decision making
- **Compliance**: Enhanced adherence to industry standards

## 🔧 Technical Details

### Data Preprocessing
- **Feature Engineering**: Created derived features like vehicle age, claim ratios
- **Encoding**: Label encoding for categorical variables
- **Scaling**: Robust scaling for numerical features
- **Imbalance Handling**: SMOTE oversampling for balanced training

### Model Training
- **80/20 Split**: Training and testing data separation
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Hyperparameter Tuning**: Grid search optimization
- **Feature Selection**: Top-K feature selection

### Evaluation Metrics
- **Confusion Matrix**: Classification performance breakdown
- **ROC Curve**: True positive vs false positive rates
- **Precision-Recall Curve**: Precision vs recall trade-offs
- **Feature Importance**: Most influential prediction factors