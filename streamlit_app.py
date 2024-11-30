import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict_heart_disease

st.title('Heart Disease Prediction Q&A')

st.write("""
This app predicts whether a person has heart disease based on their medical attributes.
Please answer the following questions:
""")

# Age
age = st.number_input('What is your age?', min_value=20, max_value=100, value=45)

# Sex
sex = st.radio('What is your gender?', ['Female', 'Male'])
sex_encoded = 1 if sex == 'Male' else 0

# Chest Pain Type
cp = st.selectbox('What type of chest pain do you experience?', 
    ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'],
    help='0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic')
cp_encoded = ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'].index(cp)

# Resting Blood Pressure
trestbps = st.number_input('What is your resting blood pressure (in mm Hg)?', 
                          min_value=90, max_value=200, value=120)

# Cholesterol
chol = st.number_input('What is your serum cholesterol level (in mg/dl)?', 
                       min_value=120, max_value=600, value=200)

# Fasting Blood Sugar
fbs = st.radio('Is your fasting blood sugar > 120 mg/dl?', ['No', 'Yes'])
fbs_encoded = 1 if fbs == 'Yes' else 0

# Resting ECG
restecg = st.selectbox('What are your resting electrocardiographic results?',
    ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
restecg_encoded = ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'].index(restecg)

# Max Heart Rate
thalach = st.number_input('What is your maximum heart rate achieved?', 
                         min_value=70, max_value=220, value=150)

# Exercise Induced Angina
exang = st.radio('Do you have exercise induced angina?', ['No', 'Yes'])
exang_encoded = 1 if exang == 'Yes' else 0

# ST Depression
oldpeak = st.number_input('ST depression induced by exercise relative to rest:', 
                         min_value=0.0, max_value=6.0, value=0.0)

# Slope of Peak Exercise ST Segment
slope = st.selectbox('What is the slope of your peak exercise ST segment?',
    ['Upsloping', 'Flat', 'Downsloping'])
slope_encoded = ['Upsloping', 'Flat', 'Downsloping'].index(slope)

# Number of Major Vessels
ca = st.radio('Number of major vessels colored by flourosopy:', 
              options=[0, 1, 2, 3], 
              horizontal=True)
# Thalassemia
thal = st.selectbox('What is your thalassemia type?',
    ['Normal', 'Fixed defect', 'Reversible defect'])
thal_encoded = ['Normal', 'Fixed defect', 'Reversible defect'].index(thal) + 1

if st.button('Predict'):
    user_data = [age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, 
                 restecg_encoded, thalach, exang_encoded, oldpeak, slope_encoded, 
                 ca, thal_encoded]
    
    prediction = predict_heart_disease(user_data)
    
    st.header('Prediction Result:')
    if prediction == 0:
        st.success('Good news! The model predicts you do not have heart disease.')
    else:
        st.warning('The model predicts you may have heart disease. Please consult a healthcare professional.')
        
    st.write('Note: This is a preliminary screening tool and should not be used as a substitute for professional medical advice.')