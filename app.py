import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Employee Productivity Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #1E293B 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 26px 32px;
        margin-bottom: 24px;
        box-shadow: 0 12px 28px -6px rgba(0, 0, 0, 0.3);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(129, 140, 248, 0.15);
        color: #A5B4FC;
        border: 1px solid rgba(129, 140, 248, 0.3);
        margin-bottom: 10px;
    }

    .score-card-high {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.14) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }

    .score-card-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.14) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }

    .score-card-low {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.14) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }

    .metric-hero {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
    }

    .insight-card {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #6366F1;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to find model and data
def get_asset_path(filename):
    # Check current dir
    if os.path.exists(filename):
        return filename
    # Check script dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p1 = os.path.join(script_dir, filename)
    if os.path.exists(p1):
        return p1
    # Check nested dir
    p2 = os.path.join(script_dir, "Employee_Productivity_Prediction_Sklearn", filename)
    if os.path.exists(p2):
        return p2
    return filename

@st.cache_resource
def load_model():
    path = get_asset_path("employee_productivity_model.pkl")
    return joblib.load(path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading productivity model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">Human Capital & Performance Intelligence</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">⚡ Employee Productivity Predictor</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Forecast employee productivity score (0 - 100) using machine learning based on workload balance, satisfaction, and working patterns.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Individual Assessment", "📁 Batch Evaluation (CSV)", "📊 Model Diagnostics & Features"])

# --- TAB 1: Individual Assessment ---
with tabs[0]:
    st.subheader("Employee Profile & Work Metrics")

    # Quick Preset Buttons
    p_cols = st.columns([1, 1, 1, 3])
    with p_cols[0]:
        load_top = st.button("🌟 Top Performer Preset", use_container_width=True)
    with p_cols[1]:
        load_burnout = st.button("⚠️ Burnout Risk Preset", use_container_width=True)
    with p_cols[2]:
        load_avg = st.button("💼 Balanced Profile", use_container_width=True)

    if load_top:
        st.session_state["exp"] = 14
        st.session_state["hours"] = 8.0
        st.session_state["tasks"] = 22
        st.session_state["meetings"] = 2
        st.session_state["breaks"] = 3
        st.session_state["training"] = 12.0
        st.session_state["sat"] = 9
        st.session_state["remote"] = "Yes"
        st.session_state["overtime"] = "No"
    elif load_burnout:
        st.session_state["exp"] = 4
        st.session_state["hours"] = 11.5
        st.session_state["tasks"] = 10
        st.session_state["meetings"] = 7
        st.session_state["breaks"] = 0
        st.session_state["training"] = 1.0
        st.session_state["sat"] = 3
        st.session_state["remote"] = "No"
        st.session_state["overtime"] = "Yes"
    elif load_avg:
        st.session_state["exp"] = 10
        st.session_state["hours"] = 7.5
        st.session_state["tasks"] = 16
        st.session_state["meetings"] = 3
        st.session_state["breaks"] = 2
        st.session_state["training"] = 6.0
        st.session_state["sat"] = 8
        st.session_state["remote"] = "Yes"
        st.session_state["overtime"] = "No"

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 🕒 Work Schedule & Activity")
        hours_worked = st.slider(
            "Hours Worked Per Day", 3.0, 14.0,
            value=float(st.session_state.get("hours", 7.5)), step=0.25,
            key="input_hours", help="Total working hours recorded per day."
        )
        tasks_completed = st.slider(
            "Tasks Completed Per Day", 0, 35,
            value=int(st.session_state.get("tasks", 16)), step=1,
            key="input_tasks", help="Number of finished work units/tickets daily."
        )
        meetings = st.slider(
            "Meetings Per Day", 0, 10,
            value=int(st.session_state.get("meetings", 3)), step=1,
            key="input_meetings", help="Frequency of synchronized meetings per day."
        )
        breaks = st.slider(
            "Breaks Taken Per Day", 0, 6,
            value=int(st.session_state.get("breaks", 2)), step=1,
            key="input_breaks", help="Rest/recharge intervals during work hours."
        )

    with col2:
        st.markdown("#### 🎯 Experience, Growth & Wellbeing")
        experience = st.slider(
            "Experience (Years)", 0, 25,
            value=int(st.session_state.get("exp", 10)), step=1,
            key="input_exp"
        )
        training = st.slider(
            "Training Hours Per Month", 0.0, 40.0,
            value=float(st.session_state.get("training", 6.0)), step=0.5,
            key="input_training"
        )
        satisfaction = st.slider(
            "Job Satisfaction Score (1 - 10)", 1, 10,
            value=int(st.session_state.get("sat", 8)), step=1,
            key="input_sat"
        )
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            remote = st.selectbox(
                "Remote Work?", ["Yes", "No"],
                index=["Yes", "No"].index(st.session_state.get("remote", "Yes")),
                key="input_remote"
            )
        with c_sub2:
            overtime = st.selectbox(
                "Regular Overtime?", ["No", "Yes"],
                index=["No", "Yes"].index(st.session_state.get("overtime", "No")),
                key="input_overtime"
            )

    st.markdown("---")

    calc_btn = st.button("🚀 Calculate Productivity Score", type="primary", use_container_width=True)

    input_df = pd.DataFrame([{
        "experience_years": experience,
        "hours_worked_per_day": hours_worked,
        "tasks_completed": tasks_completed,
        "meetings_per_day": meetings,
        "breaks_per_day": breaks,
        "training_hours_per_month": training,
        "job_satisfaction": satisfaction,
        "remote_work": remote,
        "overtime": overtime
    }])

    # Run Prediction
    pred_score = float(model.predict(input_df)[0])
    pred_score = max(0.0, min(100.0, pred_score))

    st.markdown("### 📈 Evaluation Summary")
    out_col1, out_col2 = st.columns([1.3, 1.7], gap="medium")

    with out_col1:
        if pred_score >= 85:
            card_class = "score-card-high"
            status_text = "🌟 High Productivity"
            status_color = "#10B981"
        elif pred_score >= 65:
            card_class = "score-card-medium"
            status_text = "⚡ Moderate Productivity"
            status_color = "#F59E0B"
        else:
            card_class = "score-card-low"
            status_text = "⚠️ Suboptimal / Burnout Alert"
            status_color = "#EF4444"

        st.markdown(f"""
        <div class="{card_class}">
            <div style="color: {status_color}; font-weight: 700; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px;">
                {status_text}
            </div>
            <div class="metric-hero" style="color: {status_color}; margin: 8px 0;">
                {pred_score:.1f} <span style="font-size: 1.3rem; color: #94A3B8;">/ 100</span>
            </div>
            <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Predicted Productivity Score</p>
        </div>
        """, unsafe_allow_html=True)

        st.progress(pred_score / 100.0)

    with out_col2:
        st.markdown("#### 💡 Diagnostics & Driver Feedback")
        observations = []
        if tasks_completed >= 20:
            observations.append(("High Task Velocity", "Task throughput is exceptional.", "positive"))
        elif tasks_completed < 8:
            observations.append(("Low Task Output", "Output volume is below average for the role.", "negative"))

        if meetings >= 5:
            observations.append(("Meeting Fatigue", f"{meetings} daily meetings create context switching overhead.", "negative"))

        if breaks == 0:
            observations.append(("Lack of Rest Breaks", "Zero breaks during the workday severely degrades afternoon focus.", "negative"))

        if overtime == "Yes" and hours_worked >= 10:
            observations.append(("Chronic Overtime Warning", "Long hours and overtime increase burnout probability.", "negative"))

        if satisfaction >= 8:
            observations.append(("High Morale & Engagement", "High job satisfaction correlates positively with output quality.", "positive"))
        elif satisfaction <= 4:
            observations.append(("Low Job Satisfaction", "Dissatisfaction dampens sustained motivation.", "negative"))

        if observations:
            for title, desc, tone in observations:
                if tone == "positive":
                    st.success(f"**{title}**: {desc}")
                else:
                    st.warning(f"**{title}**: {desc}")
        else:
            st.info("Performance indicators are balanced within standard operational norms.")

        st.markdown(f"""
        <div class="insight-card">
            <strong style="color: #A5B4FC;">Management Strategy Suggestion:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                {'Focus on employee retention and rewarding high contributions.' if pred_score >= 85 else ('Evaluate meeting load and encourage structured rest periods to boost engagement.' if pred_score >= 65 else 'Conduct an immediate check-in regarding workload, task clarity, and potential burnout symptoms.')}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 Inspect Input Parameters"):
        st.dataframe(input_df, use_container_width=True)

# --- TAB 2: Batch Evaluation ---
with tabs[1]:
    st.subheader("Batch Productivity Analysis")
    st.write("Upload an employee CSV or test against the bundled dataset.")

    csv_upload = st.file_uploader("Upload Employee Data CSV", type=["csv"], key="prod_csv")
    batch_df = None

    if csv_upload is not None:
        batch_df = pd.read_csv(csv_upload)
        st.info(f"Loaded {len(batch_df)} rows from uploaded file.")
    else:
        sample_path = get_asset_path("data/employee_productivity.csv")
        if os.path.exists(sample_path):
            if st.checkbox("Use baseline dataset (`data/employee_productivity.csv`)", value=True):
                batch_df = pd.read_csv(sample_path)
                st.info(f"Loaded {len(batch_df)} records from sample dataset.")

    if batch_df is not None:
        req_cols = [
            "experience_years", "hours_worked_per_day", "tasks_completed",
            "meetings_per_day", "breaks_per_day", "training_hours_per_month",
            "job_satisfaction", "remote_work", "overtime"
        ]
        missing = [c for c in req_cols if c not in batch_df.columns]
        if missing:
            st.error(f"Missing required columns in CSV: {missing}")
        else:
            if st.button("⚡ Run Batch Prediction", type="primary"):
                with st.spinner("Calculating productivity scores..."):
                    scores = model.predict(batch_df[req_cols])
                    scores = np.clip(scores, 0.0, 100.0)

                    res_df = batch_df.copy()
                    res_df["Predicted_Score"] = np.round(scores, 2)
                    res_df["Productivity_Tier"] = [
                        "High (≥85)" if s >= 85 else ("Moderate (65-84)" if s >= 65 else "Suboptimal (<65)")
                        for s in scores
                    ]

                    # Metric Summary
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Evaluated", len(res_df))
                    m2.metric("Average Score", f"{np.mean(scores):.1f} / 100")
                    m3.metric("High Performers", sum(scores >= 85))
                    m4.metric("Attention Needed", sum(scores < 65))

                    tier_filter = st.selectbox("Filter Display by Tier:", ["All", "High (≥85)", "Moderate (65-84)", "Suboptimal (<65)"])
                    if tier_filter != "All":
                        filtered_df = res_df[res_df["Productivity_Tier"] == tier_filter]
                    else:
                        filtered_df = res_df

                    st.dataframe(filtered_df, use_container_width=True)

                    csv_data = res_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="employee_productivity_predictions.csv",
                        mime="text/csv"
                    )

# --- TAB 3: Model Diagnostics ---
with tabs[2]:
    st.subheader("Model Architecture & Feature Importance")

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.markdown("""
        #### 🤖 Model Architecture
        - **Algorithm**: `RandomForestRegressor(n_estimators=200, random_state=42)`
        - **Target**: Continuous productivity score `productivity_score` ($0$ to $100$)
        - **Validation Metrics**:
            - **R² Score**: ~0.80
            - **MAE (Mean Absolute Error)**: ~4.14 points
            - **RMSE**: ~6.04 points
        - **Pipeline Preprocessor**: `ColumnTransformer` with `OneHotEncoder` for categoricals (`remote_work`, `overtime`) and passthrough for numerical metrics.
        """)

    with d_col2:
        img_path = get_asset_path("feature_importance.png")
        if os.path.exists(img_path):
            st.image(img_path, caption="Top Features Influencing Productivity", use_container_width=True)
        else:
            st.info("Feature importance image not found.")

st.caption("Employee Productivity Analytics Suite • Built with Scikit-Learn & Streamlit")
