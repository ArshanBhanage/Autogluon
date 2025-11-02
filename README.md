# AutoGluon Notebooks Summary

This repository contains three comprehensive Jupyter notebooks demonstrating AutoGluon's capabilities for automated machine learning across different problem types.

---

## 1. California Housing Price Prediction

**Notebook:** `California_Housing_Price.ipynb`

**YouTube Tutorial:** [ADD LINK HERE]

### Summary
This notebook demonstrates regression analysis using AutoGluon on the California Housing Prices dataset from Kaggle. The project walks through the complete machine learning pipeline from data acquisition to prediction generation. After downloading and preparing the housing dataset, we use AutoGluon's TabularPredictor to automatically train and optimize multiple regression models including gradient boosting machines, neural networks, and ensemble methods. The framework handles feature engineering, model selection, and hyperparameter tuning automatically with the 'good' preset configuration. The notebook evaluates model performance using root mean squared error (RMSE) and generates a submission-ready prediction file for the Kaggle competition.

**Key Features:**
- Kaggle API integration for dataset download
- Automated feature engineering and data preprocessing
- Multi-model regression training with AutoGluon
- Model leaderboard comparison
- Kaggle submission file generation

**Dataset:** California Housing Prices (Kaggle Competition)

**Problem Type:** Regression

**Evaluation Metric:** Root Mean Squared Error (RMSE)

---

## 2. Tabular and Multimodal Learning

**Notebook:** `Tabular_and_Multimodel_Atuogluon.ipynb`

**YouTube Tutorial:** [ADD LINK HERE]

### Summary
This comprehensive notebook showcases AutoGluon's versatility across three distinct use cases. First, it demonstrates the AutoMLPipelineFeatureGenerator's ability to automatically handle diverse data types including numeric, datetime, categorical, and text features, along with intelligent missing value imputation. Second, it applies TabularPredictor to the California Housing dataset, illustrating standard tabular regression with automatic model evaluation and leaderboard generation. The highlight is the third section on multimodal learning, where MNIST digit images are combined with text descriptions and statistical features to create a multimodal classification problem. This showcases AutoGluon's unique capability to simultaneously process image, text, and tabular data in a unified framework.

**Key Features:**
- Automatic feature engineering for mixed data types
- Missing value handling demonstration
- Tabular regression on California Housing dataset
- Multimodal learning combining images, text, and tabular data
- MNIST digit classification with multiple data modalities

**Datasets:** 
- Synthetic regression data
- California Housing (sklearn)
- MNIST Handwritten Digits (PyTorch)

**Problem Types:** Regression, Multimodal Classification

---

## 3. USA Rainfall Prediction

**Notebook:** `USA_Rainfall__Prediction_AutoGluon.ipynb`

**YouTube Tutorial:** [ADD LINK HERE]

### Summary
This notebook focuses on binary classification for weather prediction using AutoGluon on the USA Rainfall Prediction Dataset (2024-2025). The project predicts whether it will rain tomorrow based on various weather features including temperature, humidity, and atmospheric pressure. After downloading the dataset via KaggleHub and performing train-test split, we configure AutoGluon's TabularPredictor for binary classification with ROC AUC as the primary evaluation metric. Using the 'best_quality' preset, AutoGluon trains multiple classification models and automatically creates ensemble solutions. The notebook provides comprehensive performance evaluation including accuracy scores, ROC AUC metrics, detailed classification reports, and probability predictions for each forecast.

**Key Features:**
- KaggleHub integration for dataset access
- Binary classification for weather prediction
- ROC AUC optimization for imbalanced problems
- Comprehensive performance metrics (accuracy, precision, recall, F1-score)
- Probability predictions with confidence scores

**Dataset:** USA Rainfall Prediction Dataset 2024-2025 (Kaggle)

**Problem Type:** Binary Classification

**Evaluation Metrics:** ROC AUC, Accuracy, Precision, Recall, F1-Score

---

## About AutoGluon

AutoGluon is an open-source AutoML framework that enables developers and data scientists to achieve state-of-the-art machine learning performance with minimal code. It automatically handles:

- **Feature Engineering:** Transforms raw data into ML-ready features
- **Model Selection:** Tests multiple algorithms and architectures
- **Hyperparameter Optimization:** Fine-tunes model parameters automatically
- **Ensemble Learning:** Combines multiple models for better performance
- **Multimodal Learning:** Processes images, text, and tabular data together

These notebooks demonstrate AutoGluon's power in solving real-world problems across regression, classification, and multimodal learning tasks.

---

## Getting Started

Each notebook includes:
- ✅ Installation instructions
- ✅ Data download and preparation steps
- ✅ Model training and evaluation
- ✅ Performance visualization and interpretation
- ✅ Section headers for easy navigation

All notebooks are ready to run in Google Colab with the "Open in Colab" badge at the top of each file.

---

## Repository Structure

```
Autogluon/
├── California_Housing_Price.ipynb
├── Tabular_and_Multimodel_Atuogluon.ipynb
├── USA_Rainfall__Prediction_AutoGluon.ipynb
├── SUMMARY.md (this file)
├── Video_Scripts.md
└── README.md
```

---

*Last Updated: November 2, 2025*
