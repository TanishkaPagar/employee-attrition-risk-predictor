import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import train_model
from predict import predict_single, predict_bulk
from utils import generate_excel_report

st.set_page_config(
    page_title="Employee Attrition Risk Predictor",
    page_icon="👥",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F3864;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .low { background-color: #d4edda; color: #155724; }
    .medium { background-color: #fff3cd; color: #856404; }
    .high { background-color: #f8d7da; color: #721c24; }
    .factor-card {
        background: #f8f9fa;
        border-left: 4px solid #1F3864;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #000000;
    }
    .metric-box {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">👥 Employee Attrition Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Powered by XGBoost + OneHotEncoding + SHAP Explainability | IBM HR Analytics Dataset</div>', unsafe_allow_html=True)
st.markdown("---")

# ===================== LOAD MODEL =====================
@st.cache_resource
def get_model():
    return train_model('WA_Fn-UseC_-HR-Employee-Attrition.csv')

with st.spinner("Loading and training model..."):
    model_data = get_model()

# ===================== MODEL METRICS =====================
metrics = model_data['metrics']
cv = model_data['cv_results']
imbalance = model_data['imbalance_results']

with st.expander("📊 Model Performance Metrics", expanded=False):
    st.markdown("#### Test Set Metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    m5.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

    st.markdown("#### Stratified 5-Fold Cross Validation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV F1", f"{cv['f1']:.3f}")
    c2.metric("CV Recall", f"{cv['recall']:.3f}")
    c3.metric("CV Precision", f"{cv['precision']:.3f}")
    c4.metric("CV ROC-AUC", f"{cv['roc_auc']:.3f}")

    st.markdown("#### Confusion Matrix")
    cm = metrics['confusion_matrix']
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted: No', 'Predicted: Yes'],
        y=['Actual: No', 'Actual: Yes'],
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        showscale=False
    ))
    fig_cm.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("#### Imbalance Handling Comparison")
    imb_df = pd.DataFrame(imbalance).T.reset_index()
    imb_df.columns = ['Method', 'F1 Score', 'Recall']
    imb_df = imb_df.sort_values('F1 Score', ascending=False)
    st.dataframe(imb_df, use_container_width=True)
    st.success(f"✅ Best Method Selected: **{model_data['best_sampler']}**")

st.markdown("---")

tab1, tab2 = st.tabs(["🔍 Single Employee Prediction", "📂 Bulk Prediction (Upload CSV)"])

# ===================== TAB 1: SINGLE =====================
with tab1:
    st.subheader("Enter Employee Details")
    st.info("Fill in the details below to predict attrition risk for a single employee.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**👤 Personal Info**")
        age = st.slider("Age", 18, 65, 30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        distance = st.slider("Distance from Home (km)", 1, 30, 10)
        num_companies = st.slider("Number of Companies Worked", 0, 9, 2)

    with col2:
        st.markdown("**💼 Job Info**")
        department = st.selectbox("Department", [
            "Sales", "Research & Development", "Human Resources"
        ])
        job_role = st.selectbox("Job Role", [
            "Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"
        ])
        job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5],
                                  format_func=lambda x: {1:"Entry",2:"Junior",3:"Mid",4:"Senior",5:"Executive"}[x])
        business_travel = st.selectbox("Business Travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        overtime = st.selectbox("Works Overtime?", ["Yes", "No"])
        education_field = st.selectbox("Education Field", [
            "Life Sciences", "Medical", "Marketing",
            "Technical Degree", "Human Resources", "Other"
        ])

    with col3:
        st.markdown("**💰 Compensation & Experience**")
        monthly_income = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
        percent_hike = st.slider("Salary Hike Last Year (%)", 1, 25, 12)
        stock_option = st.selectbox("Stock Option Level", [0, 1, 2, 3])
        total_working_years = st.slider("Total Working Years", 0, 40, 8)
        years_at_company = st.slider("Years at Company", 0, 40, 5)
        years_in_role = st.slider("Years in Current Role", 0, 20, 3)
        years_since_promo = st.slider("Years Since Last Promotion", 0, 15, 2)
        training_times = st.slider("Training Times Last Year", 0, 6, 2)

    st.markdown("**⭐ Satisfaction Ratings**")
    r1, r2, r3, r4, r5 = st.columns(5)
    with r1:
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    with r2:
        work_life_balance = st.slider("Work-Life Balance", 1, 4, 3)
    with r3:
        env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 3)
    with r4:
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 3)
    with r5:
        job_involvement = st.slider("Job Involvement", 1, 4, 3)

    daily_rate = st.slider("Daily Rate", 100, 1500, 800)
    hourly_rate = st.slider("Hourly Rate", 30, 100, 65)
    monthly_rate = st.slider("Monthly Rate", 2000, 27000, 14000)

    if st.button("🔮 Predict Attrition Risk", type="primary", use_container_width=True):
        input_data = {
            'Age': age, 'BusinessTravel': business_travel,
            'DailyRate': daily_rate, 'Department': department,
            'DistanceFromHome': distance, 'EducationField': education_field,
            'EnvironmentSatisfaction': env_satisfaction, 'Gender': gender,
            'HourlyRate': hourly_rate, 'JobInvolvement': job_involvement,
            'JobLevel': job_level, 'JobRole': job_role,
            'JobSatisfaction': job_satisfaction, 'MaritalStatus': marital_status,
            'MonthlyIncome': monthly_income, 'MonthlyRate': monthly_rate,
            'NumCompaniesWorked': num_companies, 'OverTime': overtime,
            'PercentSalaryHike': percent_hike, 'RelationshipSatisfaction': relationship_satisfaction,
            'StockOptionLevel': stock_option, 'TotalWorkingYears': total_working_years,
            'TrainingTimesLastYear': training_times, 'WorkLifeBalance': work_life_balance,
            'YearsAtCompany': years_at_company, 'YearsInCurrentRole': years_in_role,
            'YearsSinceLastPromotion': years_since_promo
        }

        with st.spinner("Analyzing employee profile..."):
            prob, risk_label, factors = predict_single(input_data, model_data)

        st.markdown("---")
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            color = "#00C851" if "Low" in risk_label else "#FFD700" if "Medium" in risk_label else "#FF4444"
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob * 100, 1),
                title={'text': "Attrition Risk Score", 'font': {'size': 18}},
                number={'suffix': "%", 'font': {'size': 36}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 30], 'color': '#d4edda'},
                        {'range': [30, 60], 'color': '#fff3cd'},
                        {'range': [60, 100], 'color': '#f8d7da'},
                    ],
                }
            ))
            fig.update_layout(height=300, margin=dict(t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

            risk_class = "low" if "Low" in risk_label else "medium" if "Medium" in risk_label else "high"
            st.markdown(f'<div class="risk-card {risk_class}">{risk_label}</div>', unsafe_allow_html=True)

        with col_res2:
            st.markdown("#### 🔍 Top Factors Driving This Prediction")
            st.markdown("*(SHAP-based explainability)*")
            if factors:
                for f in factors:
                    direction = "⬆️ Increases risk" if f['impact'] > 0 else "⬇️ Decreases risk"
                    st.markdown(
                        f'<div class="factor-card"><b>{f["feature"]}</b> — {direction} '
                        f'(impact: {f["impact"]:+.3f})</div>',
                        unsafe_allow_html=True
                    )
            else:
                st.info("SHAP factors not available.")

            if "High" in risk_label:
                st.error("⚠️ High Risk! Immediate retention action recommended.")
            elif "Medium" in risk_label:
                st.warning("👀 Medium Risk. Schedule a check-in with this employee.")
            else:
                st.success("✅ Low Risk. Employee appears stable and engaged.")

# ===================== TAB 2: BULK =====================
with tab2:
    st.subheader("Bulk Employee Attrition Analysis")
    st.info("Upload a CSV with IBM HR dataset columns to score all employees at once.")

    uploaded = st.file_uploader("📂 Upload Employee CSV", type=["csv"])

    if uploaded:
        df_bulk = pd.read_csv(uploaded)
        st.success(f"✅ Loaded {len(df_bulk)} employees")
        st.dataframe(df_bulk.head(5), use_container_width=True)

        if st.button("🚀 Run Bulk Attrition Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing all employees..."):
                probs, labels = predict_bulk(df_bulk, model_data)

            df_result = df_bulk.copy()
            df_result['Risk Score (%)'] = (np.array(probs) * 100).round(1)
            df_result['Risk Level'] = labels

            st.markdown("---")
            low = sum(1 for l in labels if "Low" in l)
            med = sum(1 for l in labels if "Medium" in l)
            high = sum(1 for l in labels if "High" in l)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Employees", len(labels))
            m2.metric("🟢 Low Risk", low)
            m3.metric("🟡 Medium Risk", med)
            m4.metric("🔴 High Risk", high)

            col1, col2 = st.columns(2)
            with col1:
                fig_pie = px.pie(
                    values=[low, med, high],
                    names=["Low Risk", "Medium Risk", "High Risk"],
                    color_discrete_sequence=["#00C851", "#FFD700", "#FF4444"],
                    title="Risk Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                fig_hist = px.histogram(
                    df_result, x='Risk Score (%)',
                    nbins=20, title="Risk Score Distribution",
                    color_discrete_sequence=["#1F3864"]
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("### 📋 Full Results")
            st.dataframe(
                df_result.sort_values('Risk Score (%)', ascending=False),
                use_container_width=True
            )

            excel_buf = generate_excel_report(df_bulk, probs, labels)
            st.download_button(
                label="📥 Download Excel Report",
                data=excel_buf,
                file_name="attrition_risk_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )