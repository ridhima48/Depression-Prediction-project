
import sklearn
import streamlit as st
import pickle
import numpy as np

# Load the saved depression prediction model
with open('depression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Streamlit app UI

st.title("Depression Prediction App")
st.write("Welcome to the Depression Prediction App 😊")
st.write("This app predicts the likelihood of depression based on your inputs.")

# User input fields
with st.container():
    st.markdown('<div class="section-box">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Select your gender:", options=["Male", "Female"])
    with col2:
        age = st.number_input("Enter your age:", min_value=0, max_value=100, value=25)
    with col3:
        working_status = st.selectbox("working professional or student?", options=["Working Professional", "Student"])
    st.markdown('</div>', unsafe_allow_html=True)


horizontal_bar = "<hr style='margin-top: 0; margin-bottom: 0; height: 1px; border: 1px solid #635985;'><br>" 


dietary_habits = st.selectbox("How would you rate your dietary habits?", options=["Healthy", "Unhealthy", "Moderate"])
suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", options=["Yes", "No"])

family_history = st.selectbox("Do you have a family history of mental illness?", options=["Yes", "No"])

col4, col5, col6 = st.columns(3)
with col4:
    cgpa = st.number_input("Enter your CGPA:", min_value=0.0, max_value=10.0, value=8.0)
with col5:
    sleep_duration = st.number_input("Average hours of sleep per night:", min_value=0, max_value=24, value=7)
with col6:
    work_study_hours = st.number_input("Average work/study hours per day:", min_value=0, max_value=24, value=8)

st.markdown('<div class="section-box">', unsafe_allow_html=True)
financial_stress = st.slider("Rate your financial stress on a scale of 1-5:", 1, 5, 3)
work_pressure = st.slider("Rate your work pressure on a scale of 1-5:", 1, 5, 3)
work_satisfaction = st.slider("Rate your work satisfaction on a scale of 1-5:", 1, 5, 3)
st.markdown('</div>', unsafe_allow_html=True)

# Transform inputs into the feature vector for prediction
def transform_inputs(gender, age, working_status, cgpa, sleep_duration, dietary_habits,
                     suicidal_thoughts, work_study_hours, financial_stress, family_history,
                     work_pressure, work_satisfaction):
    # Example transformation (you'll need to adjust based on your actual model preprocessing)
    
    # Encoding categorical variables
    gender_value = 1 if gender == "Male" else 0
    working_status_value = 1 if working_status == "Working Professional" else 0
    dietary_habits_value = {"Unhealthy": 2, "Moderate": 1, "Healthy": 0}.get(dietary_habits, 1)
    suicidal_thoughts_value = 1 if suicidal_thoughts == "Yes" else 0
    family_history_value = 1 if family_history == "Yes" else 0

    work_pressure_value = work_pressure  # Already on a scale of 1-5
    financial_stress_value = financial_stress 
    work_satisfaction_value = work_satisfaction  # Already on a scale of 1-5

    # Construct the feature vector (you may need to scale or preprocess features further as needed)
    return np.array([[gender_value, age, working_status_value, cgpa, sleep_duration,
                      dietary_habits_value, suicidal_thoughts_value, work_study_hours,
                      financial_stress_value, family_history_value, work_pressure, work_satisfaction]])


if st.button("Predict Depression"):
    # Prepare input data
    input_data = transform_inputs(gender, age, working_status, cgpa, sleep_duration, dietary_habits,
                                  suicidal_thoughts, work_study_hours, financial_stress, family_history,
                                  work_pressure, work_satisfaction)
    
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    
    # Display the prediction result
    if prediction == 0:
        st.write("The model predicts: No Depression")
    else:
        st.write("The model predicts: Depression")
