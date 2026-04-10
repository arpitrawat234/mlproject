import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Boston House Price Prediction")
st.markdown("**Machine Learning App** | Support Vector Regressor (SVR)")
st.divider()

# ── Upload dataset ─────────────────────────────────────────────
st.subheader("Step 1 — Upload Dataset")
uploaded = st.file_uploader("Upload your Boston Housing CSV", type=["csv"])
st.caption("Expected columns: CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, MEDV (and optionally others)")

if not uploaded:
    st.info("👆 Upload the Boston Housing dataset CSV to get started.")
    st.stop()

# ── Load & validate ────────────────────────────────────────────
df = pd.read_csv(uploaded)

# Normalise column names
df.columns = df.columns.str.strip().str.upper()

feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS"]
target_name   = "MEDV"
required      = feature_names + [target_name]

missing_cols = [c for c in required if c not in df.columns]
if missing_cols:
    st.error(f"Missing columns in your CSV: {missing_cols}")
    st.stop()

df = df[required].dropna().astype(float)
st.success(f"Dataset loaded — {len(df)} rows, {len(required)} columns used.")

with st.expander("Preview dataset"):
    st.dataframe(df.head(10), use_container_width=True)

# ── Train model ────────────────────────────────────────────────
@st.cache_resource
def train(data_hash, dataframe):
    X = dataframe[feature_names]
    y = dataframe[target_name]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    r2     = r2_score(y_test, y_pred)
    return model, rmse, r2

with st.spinner("Training SVR model on your dataset..."):
    model, rmse, r2 = train(len(df), df)

st.divider()
st.subheader("Step 2 — Model Performance")
c1, c2 = st.columns(2)
c1.metric("R² Score", f"{r2:.4f}")
c2.metric("RMSE",     f"{rmse:.4f}")

# ── Prediction inputs ──────────────────────────────────────────
st.divider()
st.subheader("Step 3 — Enter House Features to Predict")

descriptions = [
    "Per capita crime rate by town",
    "Proportion of residential land zoned for lots over 25,000 sq ft",
    "Proportion of non-retail business acres per town",
    "Charles River dummy variable (1 if tract bounds river, else 0)",
    "Nitric oxides concentration (parts per 10 million)",
    "Average number of rooms per dwelling",
    "Proportion of owner-occupied units built prior to 1940",
    "Weighted distances to Boston employment centres",
]
defaults = df[feature_names].mean().round(2).tolist()

inputs = []
col1, col2 = st.columns(2)
for i, (feat, default, desc) in enumerate(zip(feature_names, defaults, descriptions)):
    col = col1 if i % 2 == 0 else col2
    with col:
        val = st.number_input(feat, value=default, step=0.01, format="%.4f", help=desc)
        inputs.append(val)

st.divider()

# ── Predict ────────────────────────────────────────────────────
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

    with st.expander("View input summary"):
        summary = pd.DataFrame({
            "Feature":     feature_names,
            "Value":       inputs,
            "Description": descriptions
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()
st.caption("Boston Housing Dataset · SVR Model trained on upload · Streamlit Cloud")
