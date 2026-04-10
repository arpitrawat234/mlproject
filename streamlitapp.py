import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Boston House Price Prediction")
st.markdown("**Machine Learning App** | Support Vector Regressor (SVR)")
st.divider()

# Load model
@st.cache_resource
def load_model():
    return joblib.load("svr_model.pkl")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file `svr_model.pkl` not found. Make sure it is in the same folder as this file.")
    st.stop()

# Features
feature_names  = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS"]
default_values = [3.61,   11.36, 11.14,   0.07,   0.55,  6.28, 68.57, 3.79]
descriptions   = [
    "Per capita crime rate by town",
    "Proportion of residential land zoned for lots over 25,000 sq ft",
    "Proportion of non-retail business acres per town",
    "Charles River dummy variable (1 if tract bounds river, else 0)",
    "Nitric oxides concentration (parts per 10 million)",
    "Average number of rooms per dwelling",
    "Proportion of owner-occupied units built prior to 1940",
    "Weighted distances to Boston employment centres",
]

# Input form
st.subheader("Enter House Features")

inputs = []
col1, col2 = st.columns(2)

for i, (feature, default, desc) in enumerate(zip(feature_names, default_values, descriptions)):
    col = col1 if i % 2 == 0 else col2
    with col:
        val = st.number_input(
            label=feature,
            value=default,
            step=0.01,
            format="%.4f",
            help=desc
        )
        inputs.append(val)

st.divider()

# Predict
if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
    input_array = np.array(inputs).reshape(1, -1)
    prediction  = model.predict(input_array)[0]

    st.success(f"### Predicted Price: **${prediction:,.2f}k**")

    if prediction < 15:
        st.info("💡 This indicates a relatively lower-priced area.")
    elif prediction < 35:
        st.info("💡 This indicates a mid-range neighbourhood.")
    else:
        st.info("💡 This indicates a high-value neighbourhood.")

    # Show input summary
    with st.expander("View input summary"):
        summary = pd.DataFrame({
            "Feature":     feature_names,
            "Value":       inputs,
            "Description": descriptions
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()
st.caption("Trained on the Boston Housing Dataset · SVR Model · Deployed on Streamlit Cloud")
