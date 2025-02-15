import streamlit as st
import joblib
import pandas as pd

# Load the models
lr_model = joblib.load('linear_regression_model.pkl')

# Load the encoders and scaler
label_encoders = joblib.load('label_encoders.pkl')
scaler = joblib.load('scaler.pkl')

# Get feature names from model
expected_features = joblib.load('expected_features.pkl')  # This file should contain the correct feature order

# Function to predict student performance
def predict_student_performance(model,input_data):
    input_df = pd.DataFrame([input_data])

    # Encode categorical data
    categorical_features = [
        'Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities',
        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Learning_Disabilities',
        'Parental_Education_Level', 'Distance_from_Home', 'Gender'
    ]
    
    for col, le in label_encoders.items():
        if col in categorical_features and col in input_df.columns:
            input_df[col] = le.transform([input_df[col].values[0]])[0]
    
    # Reorder columns to match expected feature order
    input_df = input_df[expected_features]
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Predict score
    predicted_score = model.predict(input_scaled)
    return predicted_score[0]

# Streamlit UI
st.title("Student Performance Prediction(beta)")

# User Inputs
hours_studied = st.number_input("Hours Studied per Week", min_value=0, max_value=100, value=10)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=75)
sleep_hours = st.number_input("Sleep Hours per Day", min_value=0, max_value=24, value=7)
tutoring_sessions = st.number_input("Tutoring Sessions per Week", min_value=0, max_value=10, value=2)
physical_activity = st.number_input("Physical Activity (hours per week)", min_value=0, max_value=6, value=2)

# Categorical Inputs
parental_involvement = st.selectbox("Parental Involvement", ['Low', 'Medium', 'High'])
access_to_resources = st.selectbox("Access to Resources", ['Low', 'Medium', 'High'])
internet_access = st.selectbox("Internet Access", ['Yes', 'No'])
teacher_quality = st.selectbox("Teacher Quality", ['Low', 'Medium', 'High'])
school_type = st.selectbox("School Type", ['Public', 'Private'])
motivation_level = st.selectbox("Motivation Level", ['Low', 'Medium', 'High'])
extracurricular_activities = st.selectbox("Extracurricular Activities", ['Yes', 'No'])
gender = st.selectbox("Gender", ['Male', 'Female'])
learning_disabilities = st.selectbox("Learning Disabilities", ['Yes', 'No'])
peer_influence = st.selectbox("Peer Influence", ['Positive', 'Negative', 'Neutral'])
parental_education = st.selectbox("Parental Education Level", ['High School', 'College', 'Postgraduate'])
distance_from_home = st.selectbox("Distance from Home", ['Near', 'Moderate', 'Far'])
family_income = st.selectbox("Family Income", ['Low', 'Medium', 'High'])

# Prediction Button
if st.button("Predict Exam Score"):
    student_data = {
        'Hours_Studied': hours_studied,
        'Attendance': attendance,
        'Previous_Scores': previous_scores,
        'Sleep_Hours': sleep_hours,
        'Tutoring_Sessions': tutoring_sessions,
        'Physical_Activity': physical_activity,
        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_to_resources,
        'Internet_Access': internet_access,
        'Teacher_Quality': teacher_quality,
        'School_Type': school_type,
        'Motivation_Level': motivation_level,
        'Extracurricular_Activities': extracurricular_activities,
        'Gender': gender,
        'Learning_Disabilities': learning_disabilities,
        'Peer_Influence': peer_influence,
        'Parental_Education_Level': parental_education,
        'Distance_from_Home': distance_from_home,
        'Family_Income': family_income
    }
    
    
    predicted_score = predict_student_performance(lr_model,student_data)
    st.success(f"Predicted Exam Score: {predicted_score:.2f}")
