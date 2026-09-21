# Employee Productivity Prediction using Python + Scikit-learn

## Project Overview
This machine learning project predicts an employee productivity score based on experience, working hours, completed tasks, meetings, breaks, training, job satisfaction, remote work, and overtime.

## Technology Used
- Python
- Pandas
- Scikit-learn
- Random Forest Regressor
- Joblib
- Matplotlib

## Files
- `data/employee_productivity.csv` - Synthetic employee dataset
- `train_model.py` - Trains and evaluates the model
- `predict.py` - Predicts productivity for a new employee
- `employee_productivity_model.pkl` - Trained model
- `feature_importance.png` - Feature importance chart
- `requirements.txt` - Required libraries

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```

### 3. Predict productivity
```bash
python predict.py
```

### 4. Run the Web Frontend
```bash
streamlit run app.py
```

## Target
`productivity_score` = predicted productivity score from 0 to 100.

## Note
The dataset is synthetic and intended for educational purposes only. Productivity predictions should not be used as the sole basis for employee evaluation or employment decisions.
