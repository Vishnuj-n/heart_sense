import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict_heart_disease, get_accuracy
from geminai import configure_genai, get_health_recommendations

# Set the title of the Streamlit app
st.title('Heart Disease Prediction - Q&A')

# Write a brief description of the app
st.write("""
This app predicts whether a person has heart disease based on their medical attributes.
Please answer the following questions:
""")

# Input field for GEMINI API Key
api_key = st.text_input('Enter your GEMINI API Key:', type='password')

# If API key is provided, configure GEMINI
if api_key:
    try:
        # Configure Gemini with provided API key
        configure_genai(api_key)
        st.success("API key configured successfully!")
    except Exception as e:
        st.error(f"Error configuring API key: {str(e)}")

# Input field for age
age = st.number_input('What is your age?', min_value=20, max_value=100, value=45)

# Input field for gender
sex = st.radio('What is your gender?', ['Female', 'Male'])
sex_encoded = 1 if sex == 'Male' else 0

# Input field for chest pain type
cp = st.selectbox('What type of chest pain do you experience?', 
    ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'],
    help='0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic')
cp_encoded = ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'].index(cp)

# Input field for resting blood pressure
trestbps = st.number_input('What is your resting blood pressure (in mm Hg)?', 
                          min_value=90, max_value=200, value=120)

# Input field for serum cholesterol level
chol = st.number_input('What is your serum cholesterol level (in mg/dl)?', 
                       min_value=120, max_value=600, value=200)

# Input field for fasting blood sugar
fbs = st.radio('Is your fasting blood sugar > 120 mg/dl?', ['No', 'Yes'])
fbs_encoded = 1 if fbs == 'Yes' else 0

# Input field for resting electrocardiographic results
restecg = st.selectbox('What are your resting electrocardiographic results?',
    ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
restecg_encoded = ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'].index(restecg)

# Input field for maximum heart rate achieved
thalach = st.number_input('What is your maximum heart rate achieved?', 
                         min_value=70, max_value=220, value=150)

# Input field for exercise induced angina
exang = st.radio('Do you have exercise induced angina?', ['No', 'Yes'])
exang_encoded = 1 if exang == 'Yes' else 0

# Input field for ST depression induced by exercise relative to rest
oldpeak = st.number_input('ST depression induced by exercise relative to rest:', 
                         min_value=0.0, max_value=6.0, value=0.0)

# Input field for slope of peak exercise ST segment
slope = st.selectbox('What is the slope of your peak exercise ST segment?',
    ['Upsloping', 'Flat', 'Downsloping'])
slope_encoded = ['Upsloping', 'Flat', 'Downsloping'].index(slope)

# Input field for number of major vessels colored by flourosopy
ca = st.radio('Number of major vessels colored by flourosopy:', 
              options=[0, 1, 2, 3], 
              horizontal=True)

# Input field for thalassemia type
thal = st.selectbox('What is your thalassemia type?',
    ['Normal', 'Fixed defect', 'Reversible defect'])
thal_encoded = ['Normal', 'Fixed defect', 'Reversible defect'].index(thal) + 1

# When the 'Predict' button is clicked
if st.button('Predict'):
    # Collect user input data
    user_data = [age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, 
                 restecg_encoded, thalach, exang_encoded, oldpeak, slope_encoded, 
                 ca, thal_encoded]
    
    # Make prediction using the collected data
    prediction = predict_heart_disease(user_data)
    accuracy = get_accuracy()
    
    # Get health recommendations based on prediction
    recommendations = get_health_recommendations(
        prediction=prediction,
        sex=sex_encoded,
        age=age,
        accuracy=accuracy
    )
    
    # Display prediction result
    st.header('Prediction Result:')
    if prediction == 0:
        st.success('Good news! The model predicts you do not have heart disease.')
    else:
        st.warning('The model predicts you may have heart disease. Please consult a healthcare professional.')
    
    # Display model accuracy and health recommendations
    st.write(f'Model Accuracy: {accuracy:.2f}%')
    st.subheader('Health Recommendations:')
    st.write(recommendations)
        
    # Disclaimer
    st.write('Note: This is a preliminary screening tool and should not be used as a substitute for professional medical advice.')
    st.write('For more information, please consult your healthcare provider.')