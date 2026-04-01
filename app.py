import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Predictor", layout="wide")

# ===== Load Assets =====
@st.cache_resource
def load_assets():
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    cols = pickle.load(open("columns.pkl", "rb"))
    data = pd.read_csv('Data.csv').head(500)  # optimized
    return model, scaler, cols, data

with st.spinner("Loading AI Model..."):
    try:
        model, scaler, model_columns, df_display = load_assets()
    except:
        st.error("⚠️ Error loading files. Run train_model.py first!")
        st.stop()

# ===== Title =====
st.title("📊 Customer Churn Analysis Dashboard")
st.caption("🚀 Built by Ashish joshi | ML Project")

# ===== Input Section =====
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("SeniorCitizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure", value=1)

with col2:
    phone = st.selectbox("PhoneService", ["No", "Yes"])
    lines = st.selectbox("MultipleLines", ["No phone service", "No", "Yes"])
    internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("OnlineSecurity", ["No", "Yes", "No internet service"])
    backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])

with col3:
    prot = st.selectbox("DeviceProtection", ["No", "Yes", "No internet service"])
    supp = st.selectbox("TechSupport", ["No", "Yes", "No internet service"])
    tv = st.selectbox("StreamingTV", ["No", "Yes", "No internet service"])
    mov = st.selectbox("StreamingMovies", ["No", "Yes", "No internet service"])
    cont = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

bill = st.selectbox("PaperlessBilling", ["Yes", "No"])
pay = st.selectbox("PaymentMethod", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

col4, col5 = st.columns(2)
with col4:
    m_charges = st.number_input("MonthlyCharges", value=29.85)
with col5:
    t_charges = st.number_input("TotalCharges", value=29.85)

# ===== Button =====
if st.button("Analyze Customer 🚀"):

    input_dict = {
        'gender': gender, 'SeniorCitizen': senior, 'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone, 'MultipleLines': lines, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup, 'DeviceProtection': prot,
        'TechSupport': supp, 'StreamingTV': tv, 'StreamingMovies': mov, 'Contract': cont,
        'PaperlessBilling': bill, 'PaymentMethod': pay,
        'MonthlyCharges': m_charges, 'TotalCharges': t_charges
    }

    # ===== Show Input =====
    st.subheader("🧾 Your Input Data")
    st.dataframe(pd.DataFrame([input_dict]))

    # ===== Preprocess =====
    input_df = pd.get_dummies(pd.DataFrame([input_dict]))
    final_input = pd.DataFrame(columns=model_columns).fillna(0)

    for c in input_df.columns:
        if c in final_input.columns:
            final_input[c] = input_df[c]

    scaled = scaler.transform(final_input)

    # ===== Prediction =====
    prediction = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    st.divider()

    # ===== Metrics =====
    c1, c2 = st.columns(2)
    c1.metric("Churn Probability", f"{prob*100:.2f}%")
    c2.metric("Status", "Churn" if prediction==1 else "Safe")

    if prediction == 1:
        st.error("⚠️ High Risk Customer")
    else:
        st.success("✅ Customer is Safe")

    # ===== Progress =====
    st.subheader("📈 Risk Level")
    st.progress(int(prob * 100))

    # ===== CENTERED SMALL GRAPHS =====
    st.subheader("📊 Visual Analysis")

    outer1, outer2, outer3 = st.columns([1,2,1])

    with outer2:
        colA, colB = st.columns(2)

        with colA:
            fig, ax = plt.subplots(figsize=(3,2))
            ax.bar(["Safe", "Churn"], [1-prob, prob])
            ax.set_ylim(0,1)
            st.pyplot(fig, use_container_width=False)

        with colB:
            fig2, ax2 = plt.subplots(figsize=(3,2))
            ax2.pie([1-prob, prob], labels=["Safe", "Churn"], autopct='%1.1f%%')
            st.pyplot(fig2, use_container_width=False)

    # ===== DATASET INSIGHT =====
    st.subheader("📈 Contract vs Churn")
    if "Churn" in df_display.columns:
        contract_churn = pd.crosstab(df_display["Contract"], df_display["Churn"])
        fig3, ax3 = plt.subplots(figsize=(4,2.5))
        contract_churn.plot(kind="bar", ax=ax3)
        st.pyplot(fig3, use_container_width=False)

    # ===== SIMILAR DATA =====
    st.subheader("📊 Similar Customers")
    st.dataframe(df_display[df_display["Contract"] == cont].head(10))

    # ===== SEARCH =====
    st.subheader("🔍 Search Customer ID")
    search = st.text_input("Enter Customer ID")

    if search:
        result = df_display[df_display["customerID"].astype(str).str.contains(search)]
        st.dataframe(result)

    # ===== FULL DATA =====
    st.subheader("📋 Full Dataset")
    st.dataframe(df_display)