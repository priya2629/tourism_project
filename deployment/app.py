
from re import S
from itertools import product
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="pavanipriyanka/tourism-project", filename="best_tourism_project_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Buyer Prediction App")
st.write("""
This application predicts whether a customer is likely to **purchase a tourism package**
Please enter the data below to get a prediction.
""")

# User input

Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
NumberOfFollowups = st.selectbox("NumberOfFollowups", ["0", "1", "2","3"])
PreferredPropertyStar = st.selectbox("PreferredPropertyStar", ["2", "3", "4","5"])
MaritalStatus = st.selectbox("MaritalStatus", ["Married", "Single","Unmarried","Divorced"])
NumberOfTrips = st.selectbox("NumberOfTrips", ["0", "1", "2","3"])
NumberOfPersonVisiting = st.selectbox("NumberOfPersonVisiting", ["0", "1", "2","3"])
PitchSatisfactionScore = st.selectbox("PitchSatisfactionScore", ["1", "2", "3","4"])
OwnCar = st.selectbox("OwnCar", ["0", "1"])
NumberOfChildrenVisiting = st.selectbox("NumberOfChildrenVisiting", ["0", "1", "2","3"])
MonthlyIncome = st.number_input("MonthlyIncome", min_value=0.0, max_value=100000.0, value=50000.0, step=100.0)
TypeofContact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("CityTier", ["1", "2", "3"])
Occupation =  st.selectbox("Occupation", ["Salaried", "Free Lancer","Small Business"])
Designation = st.selectbox("Designation", ["Manager","Senior Manager","Executive","VP","AVP"])
Gender = st.selectbox("Gender", ["Male","Female"])
ProductPitched = st.selectbox("ProductPitched", ["Deluxe","Super Deluxe","Basic","Standard","King"])


# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age ,
    'DurationOfPitch': 10.0, # Assuming a default/average value if not explicitly asked
    'NumberOfPersonVisiting': NumberOfPersonVisiting ,
    'NumberOfFollowups' : NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'Designation': Designation

}])

if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.success("The customer is **likely to purchase** the tourism package.")
    else:
        st.warning("The customer is **not likely to purchase** the tourism package.")
