import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained pipeline (scaler + model)
try:
    model = joblib.load("stacked_model_with_scaler.pkl")
except FileNotFoundError:
    st.error("❌ Model file not found. Ensure 'stacked_model_with_scaler.pkl' is available.")

# Define input features
feature_names = [
    'daily_screen_time', 'app_sessions', 'social_media_usage',
    'gaming_time', 'notifications', 'night_usage', 'age',
    'work_study_hours', 'stress_level', 'apps_installed'
]

# Streamlit UI
st.title("📱 Digital Addiction Predictor")
st.write("This app predicts whether a user is digitally addicted based on usage behavior.")

# Sidebar inputs
st.sidebar.header("📋 Input Digital Behavior")

daily_screen_time = st.sidebar.slider("Daily Screen Time (hrs)", 0.0, 16.0, 4.0, step=0.5)
app_sessions = st.sidebar.slider("App Sessions per Day", 0, 100, 30)
social_media_usage = st.sidebar.slider("Social Media Usage (hrs)", 0.0, 10.0, 2.0, step=0.5)
gaming_time = st.sidebar.slider("Gaming Time (hrs)", 0.0, 10.0, 1.0, step=0.5)
notifications = st.sidebar.slider("Notifications per Day", 0, 300, 60)
night_usage = st.sidebar.slider("Night-time Usage (hrs)", 0.0, 8.0, 1.0, step=0.5)
age = st.sidebar.slider("Age", 10, 80, 25)
work_study_hours = st.sidebar.slider("Work/Study Hours (hrs)", 0.0, 12.0, 6.0, step=0.5)
stress_level = st.sidebar.slider("Stress Level (1–10)", 1, 10, 5)
apps_installed = st.sidebar.slider("Number of Installed Apps", 0, 200, 50)

# Predict on button click
if st.sidebar.button("Predict Addiction Status"):

    # Create DataFrame from input
    input_df = pd.DataFrame([[
        daily_screen_time, app_sessions, social_media_usage, gaming_time,
        notifications, night_usage, age, work_study_hours, stress_level, apps_installed
    ]], columns=feature_names)

    # Predict using the full pipeline
    prediction = model.predict(input_df)[0]

    # Output
    st.subheader("🔍 Prediction Result")
    if prediction >= 0.5:
        st.success("🚨 The user is likely *addicted* to digital usage.")
    else:
        st.info("✅ The user is *not addicted* based on the provided behavior.")
