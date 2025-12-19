import streamlit as st
import pandas as pd
import pickle

# ---------------------------
# Load the full pipeline
# ---------------------------
model = pickle.load(open("churn_pipeline.pkl", "rb"))

# ---------------------------
# Page configuration
# ---------------------------
st.set_page_config(page_title="Customer Churn Dashboard", layout="wide")
st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("""
Predict whether a customer is likely to churn based on their profile.
Adjust the sliders and select boxes to see the prediction dynamically.
""")

# ---------------------------
# Sidebar inputs (important features)
# ---------------------------
st.sidebar.header("Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 100, 12)
MonthlyCharges = st.sidebar.slider("Monthly Charges", 0, 200, 70)
TotalCharges = st.sidebar.slider("Total Charges", 0, 10000, 840)
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])

# ---------------------------
# Fill other features with default values
# ---------------------------
input_data = {
    "gender": "Male",
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": "No",
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": InternetService,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": Contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

input_df = pd.DataFrame([input_data])

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f" Customer is likely to CHURN! ({probability*100:.2f}%)")
        else:
            st.success(f" Customer is NOT likely to churn ({probability*100:.2f}%)")

        # Display probability as a bar chart
        st.subheader("Churn Probability")
        st.bar_chart(pd.DataFrame({
            "Probability": [probability*100, 100 - probability*100]
        }, index=["Churn", "Not Churn"]))

        # Show input data
        st.subheader("Customer Input Data")
        st.dataframe(input_df, use_container_width=True)

    except ValueError as e:
        st.error(f"Prediction error: {e}")
