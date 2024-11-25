import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Sample Dataset
def load_sample_data():
    data = """No,year,month,day,hour,PM2.5,PM10,SO2,NO2,CO,O3,TEMP,PRES,DEWP,RAIN,wd,WSPM,station
    1,2013,3,1,0,4,4,4,7,300,77,-0.7,1023,-18.8,0,NNW,4.4,Aotizhongxin
    2,2013,3,1,1,8,8,4,7,300,77,-1.1,1023.2,-18.2,0,N,4.7,Aotizhongxin
    3,2013,3,1,2,7,7,5,10,300,73,-1.1,1023.5,-18.2,0,NNW,5.6,Aotizhongxin
    4,2013,3,1,3,6,6,11,11,300,72,-1.4,1024.5,-19.4,0,NW,3.1,Aotizhongxin
    5,2013,3,1,4,3,3,12,12,300,72,-2,1025.2,-19.5,0,N,2,Aotizhongxin
    6,2013,3,1,5,5,5,18,18,400,66,-2.2,1025.6,-19.6,0,N,3.7,Aotizhongxin
    7,2013,3,1,6,3,3,18,32,500,50,-2.6,1026.5,-19.1,0,NNE,2.5,Aotizhongxin"""
    return pd.read_csv(pd.compat.StringIO(data))

# Load Data
df = load_sample_data()

# Streamlit App
st.title("Data Analysis & Prediction App")
st.sidebar.title("Options")
option = st.sidebar.radio("Choose an action:", ["Dataset Summary", "Visualize Data", "Model Prediction"])

if option == "Dataset Summary":
    st.header("Dataset Summary")
    st.write("### Data Preview:")
    st.write(df.head())
    st.write("### Statistical Summary:")
    st.write(df.describe())

elif option == "Visualize Data":
    st.header("Visualize Data")
    column = st.selectbox("Select a column to visualize:", df.select_dtypes(include='number').columns)
    if st.button("Show Visualization"):
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column], kde=True, color='blue')
        plt.title(f"Distribution of {column}")
        st.pyplot(plt)

elif option == "Model Prediction":
    st.header("Model Prediction")
    
    # Machine Learning Model: Predicting PM2.5 based on TEMP, PRES, DEWP
    X = df[['TEMP', 'PRES', 'DEWP']]
    y = df['PM2.5']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Results
    st.write("### Model Results:")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")
    st.write(f"R-squared Score: {r2_score(y_test, y_pred):.2f}")
    
    st.write("### Predictions vs Actual:")
    results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
    st.write(results_df.head())

    # Visualization
    if st.button("Show Prediction Plot"):
        plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        plt.title("Actual vs Predicted")
        plt.xlabel("Actual PM2.5")
        plt.ylabel("Predicted PM2.5")
        st.pyplot(plt)
