# import streamlit as st
# import plotly.express as px
# import pandas as pd
# import pandas_profiling
#
# #import profilling capability
# from streamlit_pandas_profiling import st_profile_report
# import os
#
# from pycaret.classification import *
#
# with st.sidebar:
#     st.image("./inno2.png")
#     st.title("AutoStreamML")
#     choice = st.radio("Navigation", ["Upload", "Profiling", "ML", "Download"])
#     st.info("This application allows you to build an automated ML pipeline using Streamlit, Pandas Profiling, and PyCaret.")
#
# st.write("Hello World")
#
# if choice == "Upload":
#     st.title("Upload Your Data for Modeling")
#     file = st.file_uploader("Upload Your Dataset Here")
#     if file:
#         df = pd.read_csv(file, index_col=None)
#         st.dataframe(df)
#         st.title("Perform Machine Learning")
#         target = st.selectbox("Select the Target Column", df.columns)
#         if st.button("Start Modeling"):
#             clf = setup(data=df, target=target)
#             best_model = compare_models()
#             st.write(best_model)
#
#
#
# if choice == "Profilling":
#     st.title("Automated Exploratory Data Analysis")
#     profile_report= df.profile_report()
#     st_profile_report(profile_report)
#
# if choice == "ML":
#     pass
#
# if choice == "Download":
#     pass
#
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Discover and visualize the data to gain insights
def explore_data(data):
    st.header("Data Exploration")

    st.subheader("Preview of the Data")
    st.dataframe(data.head())

    st.subheader("Data Summary")
    st.write(data.describe())

    st.subheader("Data Visualization")
    st.write("Select columns to plot:")
    columns = st.multiselect("Select columns", data.columns)
    if columns:
        plot_data = data[columns]
        st.line_chart(plot_data)

# Step 2: Prepare the data for Machine Learning algorithms
def prepare_data(data):
    st.header("Data Preparation")

    # Handle missing values if any
    st.subheader("Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values)

    # Handle categorical variables if any
    st.subheader("Categorical Variables")
    categorical_cols = data.select_dtypes(include=["object"]).columns
    st.write(categorical_cols)

    # Handle numerical variables if any
    st.subheader("Numerical Variables")
    numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns
    st.write(numerical_cols)

    # Perform data preprocessing steps (e.g., encoding, scaling, etc.)
    # ...

# Step 3: Select a model and train it
def train_model(data, target):
    st.header("Model Training")

    # Split the data into training and testing sets
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy="mean")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    # Choose a model
    model_choice = st.selectbox("Select a model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])

    if model_choice == "Logistic Regression":
        model = LogisticRegression()
    elif model_choice == "Random Forest":
        model = RandomForestClassifier()
    elif model_choice == "Gradient Boosting":
        model = HistGradientBoostingClassifier()

    # Train the model
    model.fit(X_train, y_train)

    # Step 4: Fine-tune the model (optional)
    # ...

    # Step 5: Present the solution
    st.header("Model Evaluation")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy and display classification report
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy:", accuracy)
    st.write("Classification Report:")
    st.write(classification_report(y_test, y_pred))

def main():
    st.title("Automated ML Pipeline")

    st.sidebar.title("Navigation")
    choice = st.sidebar.selectbox("Select an option", ["Upload Data", "Explore Data", "Prepare Data", "Train Model"])
    st.sidebar.info("This application allows you to automate the machine learning pipeline.")

    if choice == "Upload Data":
        st.header("Upload Your Data")
        file = st.file_uploader("Upload an Excel or CSV file", type=["xls", "xlsx", "csv"])
        if file:
            data = pd.read_csv(file)
            st.success("Data uploaded successfully!")

            st.subheader("Preview of the Data")
            st.dataframe(data.head())

            st.session_state.data = data

    elif choice == "Explore Data":
        if "data" in st.session_state:
            explore_data(st.session_state.data)
        else:
            st.warning("Please upload the data first.")

    elif choice == "Prepare Data":
        if "data" in st.session_state:
            prepare_data(st.session_state.data)
        else:
            st.warning("Please upload the data first.")

    elif choice == "Train Model":
        if "data" in st.session_state:
            target = st.sidebar.selectbox("Select the target column", st.session_state.data.columns)
            train_model(st.session_state.data, target)
        else:
            st.warning("Please upload the data first.")

if __name__ == "__main__":
    main()

