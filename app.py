import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Boston House Price Prediction")
st.markdown("**Enhanced SVR Model** | Pipeline + Rich Visualizations")
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
df.columns = df.columns.str.strip().str.upper()

feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS"]
target_name = "MEDV"
required = feature_names + [target_name]

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
def train_model(data_hash, dataframe):
    X = dataframe[feature_names]
    y = dataframe[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, X_test, y_test, y_pred, rmse, r2, mae

with st.spinner("Training SVR model on your dataset..."):
    model, X_test, y_test, y_pred, rmse, r2, mae = train_model(len(df), df)

# ── Generate visualizations (outside tabs, cache_resource for Figure objects) ──
@st.cache_resource
def generate_visualizations():
    figs = []

    # 1. Target Distribution
    fig1, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[target_name], kde=True, color='blue', ax=ax)
    ax.set_title("Distribution of House Prices (MEDV)")
    figs.append(fig1)

    # 2. Correlation Heatmap
    corr = df.corr()[target_name].abs().sort_values(ascending=False)
    top_features = corr.index[:10]
    fig2, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[top_features].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Correlation Heatmap (with Target MEDV)")
    figs.append(fig2)

    # 3. Actual vs Predicted
    fig3, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.7, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual Prices")
    ax.set_ylabel("Predicted Prices")
    ax.set_title("Actual vs Predicted Prices")
    figs.append(fig3)

    # 4. Residual Plot
    residuals = y_test - y_pred
    fig4, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_pred, residuals, alpha=0.7, color='purple')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel("Predicted Prices")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    figs.append(fig4)

    # 5. Residual Distribution
    fig5, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color='teal', ax=ax)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residuals")
    figs.append(fig5)

    # 6. Feature Importance (approximate via absolute dual coefficients)
    svr = model.named_steps['svr']
    importance = np.abs(svr.dual_coef_).sum(axis=0)
    feat_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance[:len(feature_names)]
    }).sort_values(by='Importance', ascending=False)

    fig6, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=feat_imp, x='Importance', y='Feature', palette='viridis', ax=ax)
    ax.set_title("Feature Importance (SVR)")
    figs.append(fig6)

    return figs

st.divider()

# ── Tabs ───────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📊 Visualizations & Analysis", "ℹ️ Model Information"])

with tab1:
    st.subheader("Step 2 — Model Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("R² Score", f"{r2:.4f}")
    c2.metric("RMSE", f"{rmse:.4f}")
    c3.metric("MAE", f"{mae:.4f}")

    st.divider()
    st.subheader("Step 3 — Enter House Features")

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

    defaults = df[feature_names].mean().round(4).tolist()
    inputs = []

    col1, col2 = st.columns(2)
    for i, (feat, default, desc) in enumerate(zip(feature_names, defaults, descriptions)):
        col = col1 if i % 2 == 0 else col2
        with col:
            val = st.number_input(
                feat,
                value=default,
                step=0.01,
                format="%.4f",
                help=desc
            )
            inputs.append(val)

    if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
        input_array = np.array(inputs).reshape(1, -1)
        prediction = model.predict(input_array)[0]

        st.success(f"### Predicted Median House Price: **${prediction:,.2f}k**")

        if prediction < 15:
            st.info("💡 This indicates a relatively lower-priced area.")
        elif prediction < 35:
            st.info("💡 This indicates a mid-range neighbourhood.")
        else:
            st.info("💡 This indicates a high-value neighbourhood.")

        with st.expander("View input summary"):
            summary = pd.DataFrame({
                "Feature": feature_names,
                "Value": inputs,
                "Description": descriptions
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Exploratory Data Analysis & Model Evaluation")
    visualization_figures = generate_visualizations()

    for i, fig in enumerate(visualization_figures):
        st.pyplot(fig)
        st.caption(f"Visualization {i+1}")

with tab3:
    st.subheader("Model Information")
    st.markdown(f"""
    ### Pipeline Used
    - **Scaler**: `StandardScaler()`
    - **Model**: `SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)`

    ### Performance Metrics
    - **R² Score**: `{r2:.4f}`
    - **RMSE**: `{rmse:.4f}`
    - **MAE**: `{mae:.4f}`

    The model is trained fresh on your uploaded dataset using a scikit-learn Pipeline for reproducibility.
    """)

st.divider()
st.caption("Boston Housing Dataset · SVR Model · Streamlit App")
