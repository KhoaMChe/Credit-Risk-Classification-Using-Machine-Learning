# DASHBOARD.py

import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------
# Set Streamlit page configuration
st.set_page_config(page_title="Loan Default Risk Predictor", layout="wide")
# ---------------------------------------------

# ---------------------------------------------
# Load the trained dashboard model
with open('xgboost_dashboard_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
# ---------------------------------------------

# ---------------------------------------------
# Page Title
st.title("💼 Loan Default Risk Prediction Dashboard")

# Intro Text
st.markdown("""
Welcome to the Loan Default Risk Predictor!

This application assesses the probability that a loan applicant will default, based on key financial indicators.

Please fill out the applicant details on the left to receive a risk evaluation.
""")
# ---------------------------------------------

# Sidebar Inputs
st.sidebar.header("📋 Applicant Information")

fico_score = st.sidebar.slider("FICO Score (last_fico_range_high)", 600, 850, 700)
log_income = st.sidebar.slider("Log(Annual Income)", 8.0, 15.0, 11.0)
st.sidebar.caption("ℹ️ *Log(Annual Income) is the natural logarithm of annual income. Higher value = Higher income.*")

loan_amount = st.sidebar.slider("Loan Amount ($)", 500, 40000, 10000)

term = st.sidebar.selectbox("Loan Term (months)", [36, 60])

dti = st.sidebar.slider("Debt-to-Income Ratio (%)", 0.0, 40.0, 15.0)

# Feature array in correct order
features = np.array([[fico_score, log_income, loan_amount, term, dti]])

# ---------------------------------------------
# Predict and Display Results
st.markdown("## 🎯 Prediction Results")

if st.button("🔮 Predict Default Risk"):
    probability = xgb_model.predict_proba(features)[0][1]  # Probability of default

    st.subheader(f"**Default Probability: {round(probability*100, 2)}%**")

    # Show Risk Result
    if probability >= 0.45:
        st.error("⚠️ High Risk of Default")
    else:
        st.success("✅ Low Risk of Default")

    # ---------------------------------------------
    # Gauge Meter (Speedometer)
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Default Risk (%)"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 40], 'color': "lime"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
# ---------------------------------------------
