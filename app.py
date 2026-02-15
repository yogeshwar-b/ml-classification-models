import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

st.set_page_config(page_title="Mobile Price Predictor", layout="wide")

st.markdown("""
    <style>
    .main-header {font-size: 2.5rem; color: #2E7D32; text-align: center; font-weight: bold;}
    .sub-header {font-size: 1.2rem; color: #555; text-align: center;}
    .stButton>button {width: 100%;}
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="main-header">📱 Mobile Price Range Classifier</p>', unsafe_allow_html=True)
st.markdown("---")


try:
    scaler = joblib.load('model/scaler.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
except:
    st.error("🚨 Critical Error: Model artifacts not found. Please run 'train_model.py' first.")
    st.stop()


st.sidebar.header("⚙️ Model Configuration")

# Model Selector
model_options = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
selected_model_name = st.sidebar.selectbox("Choose Classifier", model_options)

# Load Selected Model
try:
    model = joblib.load(f"model/{selected_model_name.replace(' ', '_').lower()}.pkl")
except:
    st.error(f"Could not load {selected_model_name}.")
    st.stop()

# Batch Prediction Section (Upload)
st.sidebar.markdown("---")
st.sidebar.header("📂 Batch Prediction")

@st.cache_data
def get_github_data():
    url = "https://raw.githubusercontent.com/yogeshwar-b/ml-classification-models/main/Dataset/mobileprice.csv"
    try:
        df = pd.read_csv(url)
        return df.to_csv(index=False).encode('utf-8')
    except Exception as e:
        return None

github_csv = get_github_data()

if github_csv:
    st.sidebar.download_button(
        label="⬇Download Dataset",
        data=github_csv,
        file_name="mobileprice_github.csv",
        mime="text/csv",
        help="Download the dataset from GitHub."
    )
else:
    st.sidebar.warning("Could not fetch GitHub data.")
# ------------------------------------------------

# Download Template Button
sample_df = pd.DataFrame(columns=feature_names)
sample_csv = sample_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download CSV Template",
    data=sample_csv,
    file_name="mobile_price_template.csv",
    mime="text/csv",
    help="Use this template for batch predictions."
)

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])


if uploaded_file is not None:
    st.subheader(f"Batch Prediction Mode ({selected_model_name})")
    
    try:
        data = pd.read_csv(uploaded_file)
        
        # Check for missing columns
        missing_cols = [col for col in feature_names if col not in data.columns]
        
        if missing_cols:
            st.error(f"❌ Uploaded file is missing required columns: {missing_cols}")
        else:
            # Preprocessing
            X_input = data[feature_names]
            X_scaled = scaler.transform(X_input)
            
            # Prediction
            with st.spinner('Predicting...'):
                predictions = model.predict(X_scaled)
            
            # Map predictions
            price_map = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
            prediction_labels = [price_map[p] for p in predictions]
            
            results_df = data.copy()
            results_df['Predicted_Price_Range'] = prediction_labels
            
            # Display Results Table
            st.write("### 1. Prediction Results")
            st.dataframe(results_df.head())
            
            # Download Results Button
            csv_result = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Full Results (CSV)",
                data=csv_result,
                file_name="predicted_mobile_prices.csv",
                mime="text/csv"
            )
            
            # Evaluation Metrics
            if 'Price_Range' in data.columns:
                st.markdown("---")
                st.write("### 2. Performance Evaluation")
                
                y_true = data['Price_Range']
                
                # Metrics
                acc = accuracy_score(y_true, predictions)
                f1 = f1_score(y_true, predictions, average='weighted', zero_division=0)
                mcc = matthews_corrcoef(y_true, predictions)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Accuracy", f"{acc:.1%}")
                c2.metric("F1 Score", f"{f1:.2f}")
                c3.metric("MCC", f"{mcc:.2f}")
                
                # Confusion Matrix
                st.write("#### Confusion Matrix")
                cm = confusion_matrix(y_true, predictions)
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                            xticklabels=["Low", "Med", "High", "V.High"],
                            yticklabels=["Low", "Med", "High", "V.High"])
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)
            else:
                st.info("ℹUploaded file has no 'Price_Range' column. Metrics hidden.")
                
    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ Manual Specifications")
    
    # Manual Input Sliders
    ram = st.sidebar.slider("RAM (MB)", 256, 8000, 2048)
    battery = st.sidebar.slider("Battery Power (mAh)", 500, 6000, 1500)
    px_h = st.sidebar.slider("Pixel Height", 0, 1960, 500)
    px_w = st.sidebar.slider("Pixel Width", 0, 1998, 1000)
    int_mem = st.sidebar.slider("Internal Memory (GB)", 2, 256, 16)

    st.subheader(f"Manual Prediction Mode ({selected_model_name})")
    st.info("Adjust the sliders in the sidebar to predict the price of a single device.")

    if st.button("Predict Price Category"):
        #  Input Data
        data = {
            'Battery_Power': battery, 'Blue': 0, 'Clock_Speed': 1.5, 'Dual_SIM': 1, 'FC': 5,
            'Four_G': 1, 'Int_Memory': int_mem, 'M_Dep': 0.5, 'Mobile_Wt': 140, 'N_Cores': 4,
            'PC': 10, 'Pixel_H': px_h, 'Pixel_W': px_w, 'Ram': ram, 'Sc_H': 12, 'Sc_W': 7,
            'Talk_Time': 10, 'Three_G': 1, 'Touch_Screen': 1, 'WiFi': 1
        }
        
        
        input_df_final = pd.DataFrame(columns=feature_names)
        for col in feature_names:
            if col in data:
                input_df_final.loc[0, col] = data[col]
            else:
                input_df_final.loc[0, col] = 0 

        # Predict
        scaled_input = scaler.transform(input_df_final)
        prediction = model.predict(scaled_input)[0]
        probs = model.predict_proba(scaled_input)[0]
        
        # Display Result
        class_map = {0: "Low Cost", 1: "Medium Cost", 2: "High Cost", 3: "Very High Cost"}
        st.success(f"Predicted Category: **{class_map[prediction]}**")
        
        # Probability Chart
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