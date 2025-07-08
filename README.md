Theft Risk Classifier

Project Overview:

- This project predicts the theft risk of retail products based on features such as price, popularity, security level, and category. It uses a synthetic dataset and machine learning classification to identify products at higher risk of theft, providing actionable insights for retailers to improve security and reduce losses.
- The project includes data analysis, model training, and a Streamlit web application for interactive theft risk prediction.

Features:
- Synthetic data simulation of retail products and theft occurrences.
- Exploratory data analysis with visualizations of feature relationships and importance.
- Training and evaluation of machine learning models including Random Forest, Logistic Regression, and XGBoost.
- Interactive Streamlit app for real-time theft risk prediction based on user input.
- Alignment of input features with the trained model for consistent predictions.

Technologies Used:
- Python 3.x
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP (model interpretability)
- Matplotlib, Seaborn
- Streamlit

Usage Instructions:
Streamlit App and Model Training:

You can try the live Theft Risk Classifier app here:
https://theft-risk-classifiergit-gd4bjssmbcpekdymx2pvvq.streamlit.app/

Data Analysis and Model Evaluation (theft_model.py):

The theft_model.py script includes exploratory data analysis, model training, evaluation, and visualizations such as:
- Dataset summary statistics
- Boxplots and heatmaps showing feature correlations
- Feature importance charts
- Confusion matrix and classification reports
- Top 10 high-risk products plot

To run and view these outputs locally:
- Ensure required packages are installed (see requirements.txt).
- Run the script in a Python environment with graphical support:

python theft_model.py

- The script will display plots and print evaluation metrics in the console.

License:
- This project is licensed under the MIT License.

Author:
- Sabrina Lone
- Math and Computer Science Undergraduate Student
- GitHub: github.com/sabrinalone45
