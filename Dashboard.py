# DASHBOARD.py

import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------------
# Cấu hình trang
st.set_page_config(page_title="Dự đoán rủi ro tín dụng", layout="wide")
# ---------------------------------------------

# ---------------------------------------------
# Load model
with open('xgboost_dashboard_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
# ---------------------------------------------

# ---------------------------------------------
# Tiêu đề
st.title("💼 Hệ thống dự đoán rủi ro vỡ nợ tín dụng")

st.markdown("""
Ứng dụng này giúp dự đoán xác suất khách hàng **không trả được nợ (default)**  
dựa trên các thông tin tài chính quan trọng.

👉 Nhập thông tin khách hàng bên trái để xem mức độ rủi ro.
""")
# ---------------------------------------------

# ---------------------------------------------
# Sidebar Input
st.sidebar.header("📋 Thông tin khách hàng")

# FICO Score
fico_score = st.sidebar.slider(
    "Điểm tín dụng (FICO)",
    300, 850, 700
)

# Thu nhập
annual_income = st.sidebar.number_input(
    "Thu nhập hàng năm ($)",
    min_value=1000,
    max_value=1000000,
    value=50000,
    step=1000
)

# Loan amount
loan_amount = st.sidebar.slider(
    "Số tiền vay ($)",
    500, 40000, 10000
)

# Term
term = st.sidebar.selectbox(
    "Thời gian vay (tháng)",
    [36, 60]
)

# DTI
dti = st.sidebar.slider(
    "Tỷ lệ nợ / thu nhập (%)",
    0.0, 40.0, 15.0
)

# Interest rate
int_rate = st.sidebar.slider(
    "Lãi suất (%)",
    5.0, 30.0, 12.0
)

# ---------------------------------------------
# Tạo feature đúng thứ tự model
features = np.array([[
    int_rate,
    loan_amount,
    annual_income,
    term,
    dti,
    fico_score
]])
# ---------------------------------------------

# ---------------------------------------------
# Predict
st.markdown("## 🎯 Kết quả dự đoán")

if st.button("🔮 Dự đoán rủi ro"):

    probability = xgb_model.predict_proba(features)[0][1]

    st.subheader(f"Xác suất vỡ nợ: {round(probability * 100, 2)}%")

    # Risk label
    if probability >= 0.45:
        st.error("⚠️ Rủi ro cao (Khách hàng có khả năng vỡ nợ)")
    else:
        st.success("✅ Rủi ro thấp (Khách hàng có khả năng trả nợ tốt)")

    # ---------------------------------------------
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        title={'text': "Mức độ rủi ro (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 40], 'color': "green"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)
# ---------------------------------------------
