import streamlit as st
from predict import predict_processing_time

st.set_page_config(page_title="Visa Predictor", page_icon="🌍")

st.title("🌍 Visa Processing Time Estimator")

# Inputs
nationality = st.selectbox("Select Country", ["India", "Brazil", "Mexico"])
visa_status = st.selectbox("Visa Status", ["Approved", "Pending", "Administrative Processing"])
application_date = st.date_input("Application Date")

# Button
if st.button("Predict"):

    input_data = {
        "nationality": nationality,
        "visa_status": visa_status,
        "application_date": str(application_date)
    }

    result = predict_processing_time(input_data)

    st.success(f"✅ Estimated Processing Time: {result} days")

