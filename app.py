import streamlit as st
import pandas as pd
import pycaret.classification as pc_class
import pycaret.regression as pc_regr
import numpy as np

# File uploader
st.title("Automated Machine Learning with PyCaret")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())

    # Column selection
    all_columns = data.columns.tolist()
    drop_columns = st.multiselect("Select columns to drop", all_columns)
    if drop_columns:
        data.drop(columns=drop_columns, inplace=True)
        st.write("Updated Data Preview:", data.head())

    # EDA (Exploratory Data Analysis)
    if st.checkbox("Perform EDA?"):
        selected_columns = st.multiselect("Select columns for EDA", all_columns)
        if selected_columns:
            st.write("Descriptive Statistics:", data[selected_columns].describe())
            for column in selected_columns:
                if data[column].dtype == 'object':
                    st.write(f"Unique values in '{column}':", data[column].unique())

    # Handle missing values
    missing_option = st.radio("How do you want to handle missing values?", ("Drop rows", "Fill with mean/median/mode"))
    if missing_option == "Drop rows":
        data.dropna(inplace=True)
    else:
        for column in data.columns:
            if data[column].dtype == np.number:
                data[column].fillna(data[column].mean(), inplace=True)
            else:
                data[column].fillna(data[column].mode()[0], inplace=True)

    # Encoding categorical data
    categorical_columns = [col for col in all_columns if data[col].dtype == 'object']
    if categorical_columns:
        if st.checkbox("Encode categorical data?"):
            encoding_method = st.radio("Select encoding method", ("One-Hot Encoding", "Label Encoding"))
            if encoding_method == "One-Hot Encoding":
                data = pd.get_dummies(data, columns=categorical_columns)
            elif encoding_method == "Label Encoding":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in categorical_columns:
                    data[col] = le.fit_transform(data[col])

    # Select X and Y
    x = st.multiselect("Select features for X", data.columns.tolist())
    y = st.selectbox("Select target (Y)", data.columns.tolist())

    # Determine task type
    if data[y].dtype in [np.float64, np.int64]:
        task_type = "Regression"
        setup = pc_regr.setup(data, target=y, silent=True)
    else:
        task_type = "Classification"
        setup = pc_class.setup(data, target=y, silent=True)

    # Train models with PyCaret
    best_models = pc_regr.compare_models() if task_type == "Regression" else pc_class.compare_models()

    # Display best models
    st.write("Best Models:", best_models)

