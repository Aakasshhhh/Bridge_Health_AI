import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pandas as pd
import requests
import os
import sys
import matplotlib.pyplot as plt 

sys.path.append(os.path.join(os.path.dirname(__file__), '../src')) 
from fusion import get_bridge_health

FLASK_PREDICT_URL = "http://127.0.0.1:5000/predict"
CNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn_model.h5")

st.set_page_config(
    page_title="Bridge Health Monitor", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_cnn_model():
    try:
        return load_model(CNN_MODEL_PATH)
    except Exception:
        return None

cnn_model = load_cnn_model()

st.markdown("<h1 style='text-align: center; color: white;'> 🌉BRIDGE HEALTH MONITORING SYSTEM </h1>", unsafe_allow_html=True)
st.markdown("---")

st.sidebar.header("System Status")
if cnn_model:
    st.sidebar.success("Image CNN Model Loaded.")
else:
    st.sidebar.error("Image CNN Model Failed to Load.")
    
try:
    response = requests.get("http://127.0.0.1:5000/")
    if response.status_code == 200:
        st.sidebar.success("Vibration API (Flask) is ONLINE.")
    else:
        st.sidebar.warning("Vibration API (Flask) is responding with an error.")
except:
    st.sidebar.error("Vibration API (Flask) is OFFLINE! Run 'python app/app.py'")

col_vib, col_img = st.columns(2)

with col_vib:
    st.subheader("1. Upload Vibration Data (CSV)")
    uploaded_vibration = st.file_uploader("Vibration CSV file (200 time steps)", type=["csv"])

with col_img:
    st.subheader("2. Upload Bridge Image (Cracks)")
    uploaded_image = st.file_uploader("Bridge Image (JPG, PNG)", type=["jpg", "png"])

st.markdown("<br>", unsafe_allow_html=True) 

if uploaded_image and uploaded_vibration and cnn_model:
    
    with st.spinner("Analyzing Image for Cracks..."):
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        img_resized = cv2.resize(img, (150, 150)) / 255.0
        img_input = np.expand_dims(img_resized, axis=0)
        image_damage_conf = cnn_model.predict(img_input, verbose=0)[0][0]
        image_status = f"Damage Confidence: {image_damage_conf:.2f}"
        image_color = 'red' if image_damage_conf > 0.5 else 'green'

    with st.spinner("Sending Vibration Data to Flask API..."):
        vib_data_df = pd.read_csv(uploaded_vibration)
        if vib_data_df.empty:
            st.error("The uploaded CSV file is empty! Please upload the correct data.")
            st.stop()
            
        vib_values = vib_data_df.iloc[:, 0].values.tolist()[:200]
        
        try:
            response = requests.post(FLASK_PREDICT_URL, json={'vibration_data': vib_values})
            response.raise_for_status() 
            vib_result = response.json()
            vib_damage_conf = vib_result['damage_confidence']
            
            vib_status = f"Damage Confidence: {vib_damage_conf:.2f}"
            vib_color = 'red' if vib_damage_conf > 0.5 else 'green'
        
        except Exception:
            st.error("Error communicating with Flask API. Ensure it is running.")
            st.stop()
            
    final_health_index, final_condition = get_bridge_health(image_damage_conf, vib_damage_conf)

    st.markdown("---")
    
    st.markdown("<h2 style='text-align: center;'>FINAL STRUCTURAL INTEGRITY ASSESSMENT</h2>", unsafe_allow_html=True)
    
    col_res_l, col_res_c, col_res_r = st.columns([1, 2, 1])
    
    with col_res_c:
        if final_health_index >= 80:
            delta_text = "LOW RISK"
            delta_color = "normal"
            result_style = 'color: green; font-weight: bold;' 
        elif final_health_index >= 50:
            delta_text = "MONITORING REQUIRED"
            delta_color = "off"
            result_style = 'color: orange;'
        else:
            delta_text = "CRITICAL RISK"
            delta_color = "inverse"
            result_style = 'color: red; font-weight: bold; font-size: 24px;' 
            
        st.metric(
            label="Overall Health Index (0-100)", 
            value=f"{final_health_index}%", 
            delta=delta_text, 
            delta_color=delta_color
        )
        st.markdown(f"<p style='text-align: center; font-size: 20px; {result_style}'>Status: <b>{final_condition}</b></p>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
    st.subheader("Assessment Breakdown")
    col_break_1, col_break_2 = st.columns(2)
    
    with col_break_1:
        st.info("Image Analysis (Cracks)")
        st.image(uploaded_image, caption="AI-Detected Damage", use_container_width=True)
        st.markdown(f"**CNN Status:** <span style='color:{image_color}'>{image_status}</span>", unsafe_allow_html=True)
        
    with col_break_2:
        st.info("Vibration Analysis (LSTM Anomaly)")
        fig, ax = plt.subplots(figsize=(7, 4)) 
        ax.plot(vib_data_df.iloc[:, 0], color='gray')
        if final_health_index < 50:
             ax.plot(vib_data_df.iloc[:, 0].index, vib_data_df.iloc[:, 0], color='red', linewidth=0.5)
        
        ax.set_title("Vibration Signal Over Time (Anomaly Detection)")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)
        st.markdown(f"**LSTM Status:** <span style='color:{vib_color}'>{vib_status}</span>", unsafe_allow_html=True)