
import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Hệ thống đánh giá rủi ro tín dụng",
    layout="wide"
)

# =============================
# STYLE (BANKING UI)
# =============================
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #f8f5f0, #f1ede6);
    color: #1f2937;
}

/* Title */
.main-title {
    font-size: 28px;
    font-weight: 700;
    color: #1e3a8a;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    color: #6b7280;
    font-size: 14px;
    margin-bottom: 25px;
}

/* Card */
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 4px 12px rgba(0,0,0,0.04);
}

/* Metric */
.metric-box {
    text-align: center;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #111827;
}

.metric-label {
    font-size: 13px;
    color: #6b7280;
}

/* Risk colors */
.low {
    color: #16a34a;
}

.high {
    color: #dc2626;
}

/* Button */
.stButton>button {
    background: #1e3a8a;
    color: white;
    border-radius: 8px;
    font-weight: 600;
    height: 42px;
}

.stButton>button:hover {
    background: #1e40af;
}
</style>
""", unsafe_allow_html=True)

# =============================
# LOAD MODEL
# =============================
with open('xgboost_dashboard_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)

# =============================
# HEADER
# =============================
st.markdown('<div class="main-title">HỆ THỐNG ĐÁNH GIÁ RỦI RO TÍN DỤNG</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Phân tích khả năng vỡ nợ của khách hàng dựa trên dữ liệu tài chính</div>', unsafe_allow_html=True)

# =============================
# SIDEBAR INPUT
# =============================
st.sidebar.markdown("### Thông tin khách hàng")

fico_score = st.sidebar.slider("Điểm tín dụng (FICO)", 300, 850, 700)

annual_income = st.sidebar.number_input(
    "Thu nhập hàng năm ($)",
    min_value=1000,
    max_value=1000000,
    value=50000,
    step=1000
)

loan_amount = st.sidebar.slider("Số tiền vay ($)", 500, 40000, 10000)

term = st.sidebar.selectbox("Thời gian vay (tháng)", [36, 60])

dti = st.sidebar.slider("Tỷ lệ nợ / thu nhập (%)", 0.0, 40.0, 15.0)

int_rate = st.sidebar.slider("Lãi suất (%)", 5.0, 30.0, 12.0)

# =============================
# FEATURE
# =============================
features = np.array([[int_rate, loan_amount, annual_income, term, dti, fico_score]])

# =============================
# MAIN LAYOUT
# =============================
col1, col2 = st.columns([1, 1.2])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("#### Dữ liệu đầu vào")

    st.write(f"Lãi suất: {int_rate}%")
    st.write(f"Số tiền vay: {loan_amount}")
    st.write(f"Thu nhập: {annual_income}")
    st.write(f"Kỳ hạn: {term} tháng")
    st.write(f"DTI: {dti}%")
    st.write(f"FICO: {fico_score}")

    predict = st.button("Phân tích")

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("#### Kết quả đánh giá")

    if predict:
        probability = xgb_model.predict_proba(features)[0][1]

        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{probability:.2%}</div>
                <div class="metric-label">Xác suất vỡ nợ</div>
            </div>
            """, unsafe_allow_html=True)

        with colB:
            label = "Cao" if probability >= 0.45 else "Thấp"
            color = "high" if probability >= 0.45 else "low"

            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value {color}">{label}</div>
                <div class="metric-label">Mức rủi ro</div>
            </div>
            """, unsafe_allow_html=True)

        with colC:
            decision = "Xem xét" if probability >= 0.45 else "Chấp nhận"

            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{decision}</div>
                <div class="metric-label">Quyết định</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Mức độ rủi ro"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1e3a8a"},
                'steps': [
                    {'range': [0, 30], 'color': "#dcfce7"},
                    {'range': [30, 50], 'color': "#fef3c7"},
                    {'range': [50, 100], 'color': "#fee2e2"},
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Nhập thông tin và nhấn 'Phân tích'")

    st.markdown('</div>', unsafe_allow_html=True)

