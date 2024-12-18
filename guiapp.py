import streamlit as st
import pandas as pd

# Load the dataset
def load_data():
    data = pd.DataFrame({
        "No": [1, 2, 3, 4, 5, 6, 7, 8],
        "year": [2013, 2013, 2013, 2013, 2014, 2014, 2014, 2014],
        "month": [3, 3, 3, 3, 3, 3, 3, 3],
        "day": [1, 1, 1, 1, 1, 1, 1, 1],
        "hour": [0, 1, 2, 3, 4, 5, 6, 7],
        "PM2.5": [5, 8, 3, 5, 5, 3, 4, 3],
        "PM10": [14, 12, 6, 5, 5, 3, 4, 7],
        "SO2": [4, 6, 5, 5, 6, 13, 15, 14],
        "NO2": [12, 14, 14, 14, 21, 21, 32, 45],
        "CO": [200, 200, 200, 200, 200, 300, 300, 400],
        "O3": [85, 84, 83, 84, 77, 77, 62, 48],
        "TEMP": [-0.5, -0.7, -1.2, -1.4, -1.9, -2.4, -2.5, -1.4],
        "PRES": [1024.5, 1025.1, 1025.3, 1026.2, 1027.1, 1027.5, 1028.2, 1029.5],
        "DEWP": [-21.4, -22.1, -24.6, -25.5, -24.5, -21.3, -20.4, -20.4],
        "RAIN": [0, 0, 0, 0, 0, 0, 0, 0],
        "wd": ["NNW", "NW", "NNW", "N", "NNW", "NW", "NW", "NNW"],
        "WSPM": [5.7, 3.9, 5.3, 4.9, 3.2, 2.4, 2.2, 3.0],
        "station": ["Nongzhanguan"] * 8
    })
    return data

data = load_data()

# App title
st.title("Data Analysis App")

# Sidebar menu
option = st.sidebar.radio("Select an Option:", ["Dataset Info", "Data Visualization", "Prediction"])

if option == "Dataset Info":
    st.header("Dataset Information")
    st.write("Here are the first 10 rows of the dataset:")
    st.dataframe(data.head(10))
    st.write("Summary of dataset:")
    st.write(data.describe())

elif option == "Data Visualization":
    st.header("Data Visualization")

    # Select a column to visualize
    column = st.selectbox("Select a column to visualize:", data.columns[4:])

    st.bar_chart(data[column])

elif option == "Prediction":
    st.header("Prediction")

    # Simple prediction example: User inputs values, app predicts PM2.5 level
    st.write("Enter the following details for prediction:")

    year = st.number_input("Year:", min_value=2013, max_value=2023, value=2013)
    month = st.number_input("Month:", min_value=1, max_value=12, value=3)
    day = st.number_input("Day:", min_value=1, max_value=31, value=1)
    hour = st.number_input("Hour:", min_value=0, max_value=23, value=0)
    pm10 = st.number_input("PM10:", value=10.0)
    so2 = st.number_input("SO2:", value=5.0)

    # Basic rule-based prediction
    if st.button("Predict PM2.5"):
        predicted_pm25 = pm10 * 0.5 + so2 * 0.2  # Example rule
        st.write(f"Predicted PM2.5 level: {predicted_pm25:.2f}")