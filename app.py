import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the saved model
best_model = joblib.load('D:\web_app\pre_mod.pkl')

scaler = MinMaxScaler()

# Define the Streamlit app
st.title('Machine Learning Prediction App')

# Collect user input data for prediction
st.header('Enter Patient Data for Prediction')

Ht = st.text_input('Height (cm)', '')
Wt = st.text_input('Weight (kg)', '')
Interincisior_gap = st.text_input('Interincisior gap', '')
Sternomental_Distance = st.text_input('Sternomental Distance', '')
Thyromental_Ht = st.text_input('Thyromental Ht', '')
Neck_Circumference = st.text_input('Neck Circumference', '')
MPC = st.text_input('MPC', '')
ULBT = st.text_input('ULBT', '')
Age = st.text_input('Age', '')

# Convert input data into a dataframe
input_data = pd.DataFrame({
    'Ht (cm)': [Ht],
    'Wt': [Wt],
    'Interincisior gap': [Interincisior_gap],
    'Sternomental Distance': [Sternomental_Distance],
    'Thyromental Ht': [Thyromental_Ht],
    'Neck Circumference': [Neck_Circumference],
    'MPC': [MPC],
    'ULBT': [ULBT],
    'Age': [Age]
})



# Data Preprocessing
try:
    input_data = input_data.astype(float)
    input_data_scaled = scaler.transform(input_data)
except ValueError:
    st.error("Please enter valid numbers for all input fields.")
    st.stop()

# Make a prediction
if st.button('Predict'):
    prediction = best_model.predict(input_data_scaled)
    reverse_mapping = {0: '1', 1: '2a', 2: '2b', 3: '3a', 4: '3b'}
    result = reverse_mapping[prediction[0]]
    st.success(f'Predicted CL Grade: {result}')