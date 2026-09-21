import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("data/employee_productivity.csv")

X = df.drop("productivity_score", axis=1)
y = df["productivity_score"]

categorical_features = ["remote_work", "overtime"]
numeric_features = [
    "experience_years",
    "hours_worked_per_day",
    "tasks_completed",
    "meetings_per_day",
    "breaks_per_day",
    "training_hours_per_month",
    "job_satisfaction"
]

preprocessor = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("numeric", "passthrough", numeric_features)
])

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("Root Mean Squared Error:", round(mean_squared_error(y_test, predictions) ** 0.5, 2))
print("R2 Score:", round(r2_score(y_test, predictions), 4))

joblib.dump(pipeline, "employee_productivity_model.pkl")
print("Model saved to employee_productivity_model.pkl")

feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
importances = pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).head(15)

plt.figure(figsize=(10, 6))
plt.barh(
    importance_df["feature"][::-1],
    importance_df["importance"][::-1]
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Features for Employee Productivity Prediction")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance chart saved to feature_importance.png")
