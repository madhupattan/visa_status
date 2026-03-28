import pandas as pd
import joblib

# Load trained model
model = joblib.load("rf_model.pkl")

# Load encoders (if you saved them)
country_encoder = joblib.load("country_encoder.pkl")
visa_encoder = joblib.load("visa_encoder.pkl")

def preprocess_input(data):
    """
    Convert user input into model-ready format
    """

    df = pd.DataFrame([data])

    # Convert date
    df["application_date"] = pd.to_datetime(df["application_date"])

    # Feature Engineering
    df["month"] = df["application_date"].dt.month

    # Encode categorical variables
    df["nationality"] = country_encoder.transform(df["nationality"])
    df["visa_status"] = visa_encoder.transform(df["visa_status"])

    # Drop unused columns
    df = df[['nationality','visa_status','month']]

    return df


def predict_processing_time(input_data):
    """
    Takes user input → returns predicted processing time
    """

    processed_data = preprocess_input(input_data)

    prediction = model.predict(processed_data)

    return round(prediction[0], 2)



