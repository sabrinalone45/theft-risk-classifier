# Theft Risk Classifier
This project is a data-driven application that predicts the risk of retail product theft. It combines simulated product data with real-world NYC theft data and uses machine learning to assess risk based on product characteristics and contextual features.

# Project Overview
Retail theft is a significant concern for businesses. This project simulates a scenario where a retailer uses predictive analytics to assess which products are more likely to be stolen. The goal is to help businesses proactively adjust their security measures, pricing, or store layout based on risk levels.

# The project includes:

- A machine learning model trained on simulated retail data and real NYC theft data.
- A Streamlit app for real-time theft risk prediction with contextual features.
- A dashboard for model evaluation, including ROC curve, confusion matrix, and hyperparameter tuning.

# Features
Classifier Interface:

- Inputs include product price, popularity, category, borough, security level, stock level, discount percent, and foot traffic score.
- Outputs a binary prediction (High or Low Risk) and a confidence score.
- Highlights contributing risk factors for interpretability.
- Model Evaluation Dashboard:
- Displays ROC curve and AUC score
- Shows confusion matrix and classification report
- Performs hyperparameter tuning using GridSearchCV
- Provides cross-validation results and best parameters

# Usage Instructions
Run the app using this link!: https://theft-risk-classifiergit-gd4bjssmbcpekdymx2pvvq.streamlit.app/

# File Structure
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

# Setup Instructions
- Clone the repository:
git clone https://github.com/sabrinalone45/theft-risk-classifier.git
cd theft-risk-classifier

- Install dependencies:
pip install -r requirements.txt

- Train the model:
python train_model.py

- Run the app:
streamlit run app.py

# Dataset Access
The full NYC retail theft dataset (2016 to 2021) is not included in the repository due to file size limits. You can download it from this Dropbox link:
https://www.dropbox.com/scl/fi/w79swfb3coztbtwfwqzgq/nyc_retail_theft.csv?rlkey=b54q5jhj4znfrrmsm3b2fddv3&st=9hdelvss&dl=0

# Technologies Used
- Python (Pandas, NumPy)
- Scikit-learn
- Streamlit
- Matplotlib and Seaborn
- GridSearchCV for hyperparameter tuning

# License
This project is for educational and demonstration purposes only.

# Author
Sabrina Lone
Math and Computer Science Undergraduate Student
