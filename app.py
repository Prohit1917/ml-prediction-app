import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the saved model and scaler
best_model = joblib.load('pre_mod.pkl')
scaler = joblib.load('scaler.pkl')  # Ensure you have the scaler file 

# Define the Streamlit app
st.title('Difficult airway predictor')

# Collect user input data for prediction
st.header('Enter Patient Data for Prediction')

# Create a form to collect user input
with st.form(key='prediction_form'):
    Age = st.text_input('Age', '')
    gender = st.selectbox('Gender', ['M', 'F'])
    Ht = st.text_input('Height (cm)', '')
    Wt = st.text_input('Weight (kg)', '')
    MPC = st.text_input('MPC', '')
    Interincisior_gap = st.text_input('Interincisior gap', '')
    ULBT = st.text_input('ULBT', '')
    Sternomental_Distance = st.text_input('Sternomental Distance', '')
    Neck_Circumference = st.text_input('Neck Circumference', '')
    Thyromental_Ht = st.text_input('Thyromental Ht', '')




    
    
    submit_button = st.form_submit_button('Predict')

# Check if submit button is pressed
if submit_button:
    try:
        # Convert input data into a dataframe
        input_data = pd.DataFrame({
            'Age': [Age],
            'Ht (cm)': [Ht],
            'Wt': [Wt],
            'MPC': [MPC],
            'Interincisior gap': [Interincisior_gap],
            'ULBT': [ULBT],
            'Sternomental Distance': [Sternomental_Distance],
            'Thyromental Ht': [Thyromental_Ht],
            'Neck Circumference': [Neck_Circumference]
        })

        gender_fm = pd.DataFrame({
            'Sex' : [gender]
        }) 

        gender_Male = {
            'M': True,
            'F' : False
        }

        gender_Female = {
            'M': True,
            'F' : False
        }

        gender_fm['Male'] = gender_fm['Sex'].map(gender_Male)

        gender_fm['Female'] = gender_fm['Sex'].map(gender_Female)

        gender_fm.drop(columns=['Sex'], inplace=True)

        # Convert all columns to numeric, coercing errors to NaN
        input_data = input_data.apply(pd.to_numeric, errors='coerce')

        # Data Preprocessing
        input_data_scaled = scaler.transform(input_data)

        print(input_data_scaled, gender_fm)

        input_data_scaled_df = pd.DataFrame(input_data_scaled, columns=input_data.columns)
        
        df_combined = pd.concat([input_data_scaled_df, gender_fm], axis=1)

        # Make a prediction
        prediction = best_model.predict(df_combined)
        reverse_mapping = {0: '1', 1: '2a', 2: '2b', 3: '3a', 4: '3b'}
        result = reverse_mapping[prediction[0]]
        st.success(f'Predicted CL Grade: {result}')

        st.session_state['Ht'] = ''
        st.session_state['Wt'] = ''
        st.session_state['Interincisior_gap'] = ''
        st.session_state['Sternomental_Distance'] = ''
        st.session_state['Thyromental_Ht'] = ''
        st.session_state['Neck_Circumference'] = ''
        st.session_state['MPC'] = ''
        st.session_state['ULBT'] = ''
        st.session_state['Age'] = ''
        st.session_state['Gender'] = 'M'
        
    except ValueError:
        st.error("Please enter valid numbers for all input fields.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
