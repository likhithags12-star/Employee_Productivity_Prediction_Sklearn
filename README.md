# Employee_Productivity_Prediction_Sklearn
**Short GitHub description:**  > Machine learning project that predicts employee productivity using Random Forest, Scikit-learn, Python, and Streamlit.
# Employee Productivity Prediction using Machine Learning

A machine learning project that predicts an **employee productivity score from 0 to 100** based on work patterns, workload, job satisfaction, training, and other employee-related factors.

The project uses **Python and Scikit-learn** to train a Random Forest Regression model and provides both a command-line prediction script and an interactive **Streamlit web application**.

## Project Overview

Employee productivity can be influenced by several factors such as:

* Years of experience
* Working hours
* Tasks completed
* Number of meetings
* Break frequency
* Training hours
* Job satisfaction
* Remote work
* Overtime

This project uses these factors to predict an employee's productivity score.

The dataset is synthetic and is intended for educational and machine learning demonstration purposes.

## Features

* Employee productivity score prediction
* Random Forest Regression model
* Numerical and categorical feature preprocessing
* One-hot encoding for categorical variables
* Train/test data splitting
* Model evaluation using MAE, RMSE, and R²
* Feature importance analysis
* Command-line prediction
* Interactive Streamlit dashboard
* Individual employee assessment
* Quick employee profile presets
* Batch prediction using CSV files
* Model diagnostics and feature analysis
* Productivity score visualization

## Technology Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Regressor
* OneHotEncoder
* ColumnTransformer
* Pipeline

### Data Processing

* Pandas
* NumPy

### Model Management

* Joblib

### Visualization

* Matplotlib
* Streamlit

## Machine Learning Workflow

The project follows this workflow:

```text
Employee Dataset
       |
       v
Data Loading
       |
       v
Feature Selection
       |
       v
Data Preprocessing
       |
       +----------------------+
       |                      |
       v                      v
Numerical Features      Categorical Features
       |                      |
       |                One-Hot Encoding
       |                      |
       +----------+-----------+
                  |
                  v
        Random Forest Regressor
                  |
                  v
          Productivity Score
                  |
                  v
        Model Evaluation
```

## Dataset

The dataset is located at:

```text
data/employee_productivity.csv
```

The dataset contains the following features:

| Feature                    | Description                             |
| -------------------------- | --------------------------------------- |
| `experience_years`         | Employee experience in years            |
| `hours_worked_per_day`     | Average working hours per day           |
| `tasks_completed`          | Number of tasks completed per day       |
| `meetings_per_day`         | Number of meetings attended per day     |
| `breaks_per_day`           | Number of breaks taken per day          |
| `training_hours_per_month` | Training hours completed per month      |
| `job_satisfaction`         | Job satisfaction score                  |
| `remote_work`              | Whether the employee works remotely     |
| `overtime`                 | Whether the employee works overtime     |
| `productivity_score`       | Target productivity score from 0 to 100 |

## Model

The project uses a:

**Random Forest Regressor**

The model is configured with:

```python
RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
```

The data is divided into training and testing sets using an 80/20 split.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## Preprocessing

The project uses a Scikit-learn `Pipeline` and `ColumnTransformer`.

### Numerical Features

The following features are passed directly to the model:

```text
experience_years
hours_worked_per_day
tasks_completed
meetings_per_day
breaks_per_day
training_hours_per_month
job_satisfaction
```

### Categorical Features

The following features are converted using One-Hot Encoding:

```text
remote_work
overtime
```

Unknown categories are handled safely using:

```python
OneHotEncoder(handle_unknown="ignore")
```

## Model Evaluation

The trained model is evaluated using three regression metrics:

### Mean Absolute Error

MAE measures the average absolute difference between the actual and predicted productivity scores.

### Root Mean Squared Error

RMSE measures prediction error while giving greater weight to larger errors.

### R² Score

R² indicates how well the model explains the variation in the target productivity scores.

The training script automatically calculates these metrics when the model is trained.

## Feature Importance

The project generates a feature importance chart using the Random Forest model.

The output is saved as:

```text
feature_importance.png
```

This helps identify which input features contribute most to the model's predictions.

## Streamlit Application

The project includes an interactive Streamlit dashboard.

Run it using:

```bash
streamlit run app.py
```

The application contains three main sections.

### 1. Individual Assessment

Users can enter employee information such as:

* Experience
* Working hours
* Tasks completed
* Meetings
* Breaks
* Training hours
* Job satisfaction
* Remote work
* Overtime

The application then displays the predicted productivity score.

### 2. Quick Presets

The application includes example profiles:

* Top Performer
* Burnout Risk
* Balanced Profile

These presets make it easy to test the model with predefined employee profiles.

### 3. Batch Evaluation

The application supports CSV-based employee evaluation, allowing multiple employee records to be processed together.

### 4. Model Diagnostics & Features

The dashboard also provides model-related information and feature analysis to help understand the prediction system.

## Command-Line Prediction

The project also provides a simple command-line prediction script.

Run:

```bash
python predict.py
```

The program asks for employee information:

```text
Experience in years
Hours worked per day
Tasks completed per day
Meetings per day
Breaks per day
Training hours per month
Job satisfaction score
Remote work
Overtime
```

It then displays:

```text
Predicted Employee Productivity Score: XX.XX/100
```

## Project Structure

```text
Employee_Productivity_Prediction_Sklearn/
│
├── app.py
│
├── data/
│   └── employee_productivity.csv
│
├── employee_productivity_model.pkl
│
├── feature_importance.png
│
├── predict.py
│
├── train_model.py
│
├── requirements.txt
│
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd Employee_Productivity_Prediction_Sklearn
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Train the Model

Run:

```bash
python train_model.py
```

This will:

1. Load the employee dataset.
2. Separate input features and target values.
3. Preprocess categorical features.
4. Split the dataset into training and testing data.
5. Train the Random Forest Regression model.
6. Evaluate the model.
7. Save the trained model.
8. Generate the feature importance chart.

The trained model is saved as:

```text
employee_productivity_model.pkl
```

## Run Prediction

For command-line prediction:

```bash
python predict.py
```

## Run the Web Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Requirements

The project requires:

```text
pandas
scikit-learn
joblib
matplotlib
streamlit
```

These dependencies are also available in:

```text
requirements.txt
```

## Example Prediction Inputs

Example employee profile:

```text
Experience: 10 years
Hours worked: 7.5 hours/day
Tasks completed: 16/day
Meetings: 3/day
Breaks: 2/day
Training: 6 hours/month
Job satisfaction: 8/10
Remote work: Yes
Overtime: No
```

The model uses these inputs to generate a productivity score between 0 and 100.

## Important Note

The dataset used in this project is **synthetic** and is intended for educational and demonstration purposes.

The predicted productivity score should not be used as the sole basis for employee evaluation, hiring, promotion, compensation, disciplinary action, or other employment decisions.

## Future Improvements

Possible improvements include:

* Add cross-validation
* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Compare Random Forest with other regression models
* Add model performance visualizations
* Add prediction confidence or uncertainty estimates
* Improve batch CSV validation
* Add database integration
* Deploy the Streamlit application online
* Add authentication
* Add downloadable prediction reports
* Use a larger real-world dataset where appropriate
* Add explainable-AI techniques such as SHAP

## Learning Outcomes

This project demonstrates practical implementation of:

* Data preprocessing
* Feature engineering
* Categorical encoding
* Machine learning pipelines
* Random Forest Regression
* Regression model evaluation
* Feature importance
* Model serialization with Joblib
* Interactive ML applications with Streamlit
  
