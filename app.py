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

st.title("Boston House Price Prediction")
st.write("This app predicts house prices using a Support Vector Regression (SVR) model.")

# upload csv file
st.subheader("Upload Dataset")
uploaded = st.file_uploader("Upload Boston Housing CSV file", type=["csv"])

if not uploaded:
    st.warning("Please upload the dataset to continue.")
    st.stop()

# load the data
df = pd.read_csv(uploaded)
df.columns = df.columns.str.strip().str.upper()

st.write("Dataset loaded successfully!")
st.write(df.head())

# define features and target
feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS"]
target_name = "MEDV"

X = df[feature_names]
y = df[target_name]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# train model
st.subheader("Training the Model...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# show results
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

st.subheader("Model Performance")
st.write("R2 Score:", round(r2, 4))
st.write("RMSE:", round(rmse, 4))
st.write("MAE:", round(mae, 4))

# some plots
st.subheader("Visualizations")

# plot 1 - price distribution
fig1, ax1 = plt.subplots()
ax1.hist(df[target_name], bins=30, color="steelblue")
ax1.set_title("House Price Distribution")
ax1.set_xlabel("Price")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# plot 2 - actual vs predicted
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, color="blue", alpha=0.5)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax2.set_xlabel("Actual Price")
ax2.set_ylabel("Predicted Price")
ax2.set_title("Actual vs Predicted")
st.pyplot(fig2)

# plot 3 - correlation heatmap
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
ax3.set_title("Correlation Heatmap")
st.pyplot(fig3)

# predict house price
st.subheader("Predict House Price")
st.write("Enter the details below to get a price prediction.")

crim = st.number_input("Crime Rate (CRIM)", value=float(df["CRIM"].mean()))
zn = st.number_input("Residential Land Zone (ZN)", value=float(df["ZN"].mean()))
indus = st.number_input("Industrial Area (INDUS)", value=float(df["INDUS"].mean()))
chas = st.number_input("Near Charles River? (CHAS) - 1 for Yes, 0 for No", value=0.0)
nox = st.number_input("Nitric Oxide Concentration (NOX)", value=float(df["NOX"].mean()))
rm = st.number_input("Average Rooms (RM)", value=float(df["RM"].mean()))
age = st.number_input("Old Houses % (AGE)", value=float(df["AGE"].mean()))
dis = st.number_input("Distance to Employment Centers (DIS)", value=float(df["DIS"].mean()))

if st.button("Predict Price"):
    input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis]])
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted House Price: ${prediction:.2f}k")
