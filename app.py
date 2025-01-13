import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Load California Housing Dataset
california = fetch_california_housing(as_frame=True)
data = california.frame

# Sidebar inputs
st.sidebar.header("User Input Features")
MedInc = st.sidebar.slider("Median Income", float(data.MedInc.min()), float(data.MedInc.max()), float(data.MedInc.mean()))
HouseAge = st.sidebar.slider("House Age", float(data.HouseAge.min()), float(data.HouseAge.max()), float(data.HouseAge.mean()))
AveRooms = st.sidebar.slider("Average Rooms", float(data.AveRooms.min()), float(data.AveRooms.max()), float(data.AveRooms.mean()))
AveOccup = st.sidebar.slider("Average Occupancy", float(data.AveOccup.min()), float(data.AveOccup.max()), float(data.AveOccup.mean()))

# Train Linear Regression Model
X = data[['MedInc', 'HouseAge', 'AveRooms', 'AveOccup']]
y = data['MedHouseVal']
model = LinearRegression()
model.fit(X, y)

# Make predictions
input_features = pd.DataFrame([[MedInc, HouseAge, AveRooms, AveOccup]], 
                               columns=['MedInc', 'HouseAge', 'AveRooms', 'AveOccup'])
prediction = model.predict(input_features)

# Display results
st.title("House Price Prediction App")
st.write("This app predicts the median house value based on user inputs.")
st.write(f"**Predicted House Price:** ${prediction[0] * 1000:.2f}")

