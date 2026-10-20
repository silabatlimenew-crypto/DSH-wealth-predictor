# DHS Household Wealth Index Predictor 🇪🇹

An end-to-end Machine Learning web application built using **Google Colab**, **Scikit-Learn**, and **Streamlit** to predict household wealth categories (**Poor**, **Middle**, **Rich**) based on DHS (Demographic and Health Surveys) data from the Abbay Basin.

## 📌 Project Overview
The **Household Wealth Index** is a crucial metric used by governments, researchers, and NGOs to measure socioeconomic status and allocate public health and development resources. This project utilizes machine learning to predict household wealth levels using essential asset, housing, and geographic location features.

## 📊 Model Performance
Multiple machine learning algorithms were trained and evaluated, with **Random Forest Classifier** achieving the highest accuracy:
- **Model Used:** Random Forest Classifier
- **Overall Accuracy:** `81.46%`
- **Target Classes:** 
  - `Poor` (Poorest + Poorer)
  - `Middle` (Middle)
  - `Rich` (Richer + Richest)

## 🛠️ Project Structure
```text
├── app.py                   # Streamlit web application interface
├── best_wealth_model.pkl    # Trained Random Forest model file
├── selected_features.pkl    # Preprocessed feature columns
├── requirements.txt         # Required Python dependencies
└── README.md                # Project documentation
