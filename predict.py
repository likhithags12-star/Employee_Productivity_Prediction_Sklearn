import joblib
import pandas as pd

model = joblib.load("employee_productivity_model.pkl")

print("Enter employee details")

experience = int(input("Experience in years: "))
hours_worked = float(input("Hours worked per day: "))
tasks_completed = int(input("Tasks completed per day: "))
meetings = int(input("Meetings per day: "))
breaks = int(input("Breaks per day: "))
training_hours = float(input("Training hours per month: "))
satisfaction = int(input("Job satisfaction score (1-10): "))
remote = input("Remote work? (Yes/No): ")
overtime = input("Overtime? (Yes/No): ")

data = pd.DataFrame([{
    "experience_years": experience,
    "hours_worked_per_day": hours_worked,
    "tasks_completed": tasks_completed,
    "meetings_per_day": meetings,
    "breaks_per_day": breaks,
    "training_hours_per_month": training_hours,
    "job_satisfaction": satisfaction,
    "remote_work": remote,
    "overtime": overtime
}])

prediction = model.predict(data)[0]
print(f"\nPredicted Employee Productivity Score: {prediction:.2f}/100")
