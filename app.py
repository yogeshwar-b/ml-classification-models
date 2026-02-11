import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Mobile Price Predictor", layout="wide")
st.title(" Mobile Price Range Classifier")
st.markdown("Predicts: **Low Cost (0), Medium Cost (1), High Cost (2), Very High Cost (3)**")

# Load Artifacts
try:
    scaler = joblib.load('model/scaler.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
except:
    st.error("Model artifacts not found. Please run 'train_model.py' first.")
    st.stop()

# Sidebar
st.sidebar.header("Model Configuration")
model_options = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
selected_model_name = st.sidebar.selectbox("Choose Classifier", model_options)

# Load Model
try:
    model = joblib.load(f"model/{selected_model_name.replace(' ', '_').lower()}.pkl")
except:
    st.error(f"Could not load {selected_model_name}.")
    st.stop()

#  Input Features
st.sidebar.header("Device Specifications")
def user_input_features():
    ram = st.sidebar.slider("RAM (MB)", 256, 8000, 2048)
    battery = st.sidebar.slider("Battery Power (mAh)", 500, 6000, 1500)
    px_h = st.sidebar.slider("Pixel Height", 0, 1960, 500)
    px_w = st.sidebar.slider("Pixel Width", 0, 1998, 1000)
    int_mem = st.sidebar.slider("Internal Memory (GB)", 2, 256, 16)
    
    data = {
        'Battery_Power': battery, 'Blue': 0, 'Clock_Speed': 1.5, 'Dual_SIM': 1, 'FC': 5,
        'Four_G': 1, 'Int_Memory': int_mem, 'M_Dep': 0.5, 'Mobile_Wt': 140, 'N_Cores': 4,
        'PC': 10, 'Pixel_H': px_h, 'Pixel_W': px_w, 'Ram': ram, 'Sc_H': 12, 'Sc_W': 7,
        'Talk_Time': 10, 'Three_G': 1, 'Touch_Screen': 1, 'WiFi': 1
    }
    features = pd.DataFrame([data])
    final_df = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in data:
            final_df.loc[0, col] = data[col]
        else:
            final_df.loc[0, col] = 0 
    return final_df

input_df = user_input_features()

# Main Page
if st.button("Predict Price Category"):
    input_df_final = pd.DataFrame(columns=feature_names)
    for col in feature_names:
        if col in input_df.columns:
            input_df_final.loc[0, col] = input_df.loc[0, col]
        else:
            input_df_final.loc[0, col] = 0 # Handle missing columns safely

    scaled_input = scaler.transform(input_df_final)
    
    # Predict
    prediction = model.predict(scaled_input)[0]
    probs = model.predict_proba(scaled_input)[0]
    
    class_map = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
    st.success(f" Predicted Category: **{class_map[prediction]}**")
    
    prob_df = pd.DataFrame({
        'Price Range': ["Low Cost", "Medium Cost", "High Cost", "Very High Cost"],
        'Probability': probs
    })
    
    prob_df['Price Range'] = pd.Categorical(
        prob_df['Price Range'], 
        categories=["Low Cost", "Medium Cost", "High Cost", "Very High Cost"], 
        ordered=True
    )
    prob_df = prob_df.sort_values('Price Range')
    
    st.bar_chart(prob_df.set_index('Price Range'))