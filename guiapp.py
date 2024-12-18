import time  # to simulate real-time data, time loop
import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import streamlit as st  # 🎈 data web app development

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="wide",
)

# Load dataset
dataset_url = "https://github.com/SKANALYST/blank-app/blob/main/PRAC%20DATA1.csv"  # Replace with your dataset URL

@st.experimental_memo
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df = get_data()

# Sidebar navigation buttons
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to", ["Dataset Info", "Data Visualization", "Prediction"]
)

# Page: Dataset Info
if page == "Dataset Info":
    st.title("Dataset Information")
    st.write("### Overview")
    st.write(df.describe())
    st.write("### Raw Data")
    st.dataframe(df)

# Page: Data Visualization
elif page == "Data Visualization":
    st.title("Data Visualization")
    
    station_filter = st.selectbox("Select the Station", pd.unique(df["station"]))
    df_filtered = df[df["station"] == station_filter]

    # Add computed columns
    df_filtered["PM2.5_new"] = df_filtered["PM2.5"] * np.random.choice(range(1, 3))
    df_filtered["TEMP_new"] = df_filtered["TEMP"] + np.random.uniform(-5, 5)

    st.write("### PM2.5 Distribution by Hour")
    st.line_chart(df_filtered[["hour", "PM2.5_new"]].set_index("hour"))

    st.write("### Temperature Distribution")
    st.bar_chart(df_filtered[["hour", "TEMP_new"]].set_index("hour"))

# Page: Prediction
elif page == "Prediction":
    st.title("Prediction")
    st.write("### Future Features for Predictions")
    st.write("This section will be updated with prediction capabilities.")
