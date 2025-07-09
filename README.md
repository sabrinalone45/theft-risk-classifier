# Theft Risk Classifier

This project is a data-driven application that predicts the risk of retail product theft. It combines simulated product data with real-world NYC theft data and uses machine learning to assess risk based on product characteristics and contextual features.

## Project Overview

Retail theft is a significant concern for businesses. This project simulates a scenario where a retailer uses predictive analytics to assess which products are more likely to be stolen. The goal is to help businesses proactively adjust their security measures, pricing, or store layout based on risk levels.

The project includes:

- A machine learning model trained on simulated retail data and real NYC theft data.
- A Streamlit app for real-time theft risk prediction with contextual features.
- A dashboard for model evaluation, including ROC curve, confusion matrix, and hyperparameter tuning.

## Features

### Classifier Interface
- Inputs: product price, popularity, category, borough, security level, stock level, discount percent, and foot traffic score.
- Outputs: binary prediction (High or Low Risk) and a confidence score.
- Risk reasoning: highlights contributing factors for interpretability.

### Model Evaluation Dashboard
- ROC curve and AUC score
- Confusion matrix and classification report
- GridSearchCV-based hyperparameter tuning
- Cross-validation results and best parameters

## File Structure

theft-risk-classifier/
├── app.py
├── classifier_ui.py
├── model_dashboard.py
├── train_model.py
├── rf_model.pkl
├── model_training_data.pkl
├── requirements.txt
├── README.md
└── data/
└── nyc_retail_theft_sample.csv

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/theft-risk-classifier.git
cd theft-risk-classifier
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Train the model:

bash
Copy
Edit
python train_model.py
Run the app:

bash
Copy
Edit
streamlit run app.py
Dataset Access
The full NYC retail theft dataset (2016–2021) is not included in the repository due to size limits. You can download it from the following Google Drive link:

https://www.dropbox.com/scl/fi/w79swfb3coztbtwfwqzgq/nyc_retail_theft.csv?rlkey=b54q5jhj4znfrrmsm3b2fddv3&st=9hdelvss&dl=0

Alternatively, a smaller sample (nyc_retail_theft_sample.csv) is included in the data/ folder for demo and testing purposes.

Technologies Used
Python (Pandas, NumPy)

Scikit-learn

Streamlit

Matplotlib & Seaborn

GridSearchCV

Future Improvements
Deploy app via Streamlit Cloud or Hugging Face Spaces

Add SHAP or LIME explainability

Enable geospatial and time-based analysis

Implement user authentication and logging

License
This project is for educational and demonstration purposes only.

yaml
Copy
Edit
