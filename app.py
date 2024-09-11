import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the saved model and scaler
best_model = joblib.load('pre_mod.pkl')
scaler = joblib.load('scaler.pkl')  # Load scaler if you saved it

# Define the Streamlit app
st.title('Machine Learning Prediction App')

# Collect user input data for prediction
st.header('Enter Patient Data for Prediction')

# Create a form to collect user input
with st.form(key='prediction_form'):
    Ht = st.text_input('Height (cm)', '')
    Wt = st.text_input('Weight (kg)', '')
    Interincisior_gap = st.text_input('Interincisior gap', '')
    Sternomental_Distance = st.text_input('Sternomental Distance', '')
    Thyromental_Ht = st.text_input('Thyromental Ht', '')
    Neck_Circumference = st.text_input('Neck Circumference', '')
    MPC = st.text_input('MPC', '')
    ULBT = st.text_input('ULBT', '')
    Age = st.text_input('Age', '')
    
    submit_button = st.form_submit_button('Predict')

    # Check if submit button is pressed
    if submit_button:
        # Convert input data into a dataframe
        try:
            input_data = pd.DataFrame({
                'Ht (cm)': [float(Ht)],
                'Wt': [float(Wt)],
                'Interincisior gap': [float(Interincisior_gap)],
                'Sternomental Distance': [float(Sternomental_Distance)],
                'Thyromental Ht': [float(Thyromental_Ht)],
                'Neck Circumference': [float(Neck_Circumference)],
                'MPC': [float(MPC)],
                'ULBT': [float(ULBT)],
                'Age': [float(Age)]
            })

            # Data Preprocessing
            input_data_scaled = scaler.transform(input_data)
            
            # Make a prediction
            prediction = best_model.predict(input_data_scaled)
            reverse_mapping = {0: '1', 1: '2a', 2: '2b', 3: '3a', 4: '3b'}
            result = reverse_mapping[prediction[0]]
            st.success(f'Predicted CL Grade: {result}')
        except ValueError:
            st.error("Please enter valid numbers for all input fields.")