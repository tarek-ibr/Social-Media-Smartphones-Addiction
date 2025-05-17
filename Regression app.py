import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load scalers and model
try:
    x_scaler = joblib.load('regression_x_scaler.pkl')      # scaler for X
    y_scaler = joblib.load('regression_y_scaler.pkl')    # scaler for y
    model = joblib.load('linear_regression_model.pkl')  # your trained regression model
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Please ensure 'scaler.pkl', 'y_scaler.pkl', and 'linear_regression_model.pkl' exist.")
    st.stop()

st.title("🛌 Sleep Efficiency Predictor")
st.write("Predict your Sleep Efficiency (%) based on social media use and sleep-related behaviors.")

# Numeric inputs
daily_sm_time = st.slider('Average Daily Social Media Use Time (minutes)', 0, 600, 120)
freq_sm_check = st.slider('Frequency of Social Media Checking (number of times per day)', 0, 50, 10)
pre_sleep_sm_duration = st.slider('Pre-Sleep Social Media Use Duration (minutes)', 0, 180, 30)
blue_light_exposure = st.slider('Blue Light Exposure Before Sleep (minutes)', 0, 180, 30)

# Categorical inputs limited to your dataset categories
dominant_platform = st.selectbox(
    'Dominant Social Media Platform',
    ['Twitter', 'TikTok', 'Instagram', 'Facebook', 'Snapchat']
)

content_type = st.selectbox(
    'Type of Social Media Content Consumed',
    ['News', 'Social Interaction', 'Entertainment']
)

# Prepare input DataFrame
input_dict = {
    'Average Daily Social Media Use Time (minutes)': daily_sm_time,
    'Frequency of Social Media Checking (number of times per day)': freq_sm_check,
    'Pre-Sleep Social Media Use Duration (minutes)': pre_sleep_sm_duration,
    'Blue Light Exposure Before Sleep (minutes)': blue_light_exposure,
    'Dominant Social Media Platform': dominant_platform,
    'Type of Social Media Content Consumed': content_type
}
input_df = pd.DataFrame([input_dict])

# Dummy encode categorical features (drop_first to match training)
categorical_cols = ['Dominant Social Media Platform', 'Type of Social Media Content Consumed']
input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

# Align input features to model's expected features
model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_df.columns.tolist()
for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[model_features]

# Numerical columns for scaling
num_cols = [
    'Average Daily Social Media Use Time (minutes)',
    'Frequency of Social Media Checking (number of times per day)',
    'Pre-Sleep Social Media Use Duration (minutes)',
    'Blue Light Exposure Before Sleep (minutes)'
]

# Scale numeric features
input_df.loc[:, num_cols] = x_scaler.transform(input_df[num_cols])

if st.button("Predict Sleep Efficiency"):
    # Predict scaled target
    y_pred_scaled = model.predict(input_df)
    # Inverse transform prediction to original scale
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()[0]
    st.subheader("🛏️ Predicted Sleep Efficiency (%)")
    st.write(f"{y_pred:.2f} %")