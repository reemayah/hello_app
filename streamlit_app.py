{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNVFU6i7tKqjXsB1WcqZO15",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/reemayah/hello_app/blob/main/streamlit_app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "DAjEubHEeFR0"
      },
      "outputs": [],
      "source": [
        "pip install pycaret"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit"
      ],
      "metadata": {
        "id": "Vek7jaKMgzxk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile streamlit_app.py\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import pycaret.classification as pc_class\n",
        "import pycaret.regression as pc_regr\n",
        "import numpy as np\n",
        "\n",
        "# File uploader\n",
        "st.title(\"Automated Machine Learning with PyCaret\")\n",
        "uploaded_file = st.file_uploader(\"Upload your CSV file\", type=[\"csv\"])\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    data = pd.read_csv(uploaded_file)\n",
        "    st.write(\"Data Preview:\", data.head())\n",
        "\n",
        "    # Column selection\n",
        "    all_columns = data.columns.tolist()\n",
        "    drop_columns = st.multiselect(\"Select columns to drop\", all_columns)\n",
        "    if drop_columns:\n",
        "        data.drop(columns=drop_columns, inplace=True)\n",
        "        st.write(\"Updated Data Preview:\", data.head())\n",
        "\n",
        "    # EDA (Exploratory Data Analysis)\n",
        "    if st.checkbox(\"Perform EDA?\"):\n",
        "        selected_columns = st.multiselect(\"Select columns for EDA\", all_columns)\n",
        "        if selected_columns:\n",
        "            st.write(\"Descriptive Statistics:\", data[selected_columns].describe())\n",
        "            for column in selected_columns:\n",
        "                if data[column].dtype == 'object':\n",
        "                    st.write(f\"Unique values in '{column}':\", data[column].unique())\n",
        "\n",
        "    # Handle missing values\n",
        "    missing_option = st.radio(\"How do you want to handle missing values?\", (\"Drop rows\", \"Fill with mean/median/mode\"))\n",
        "    if missing_option == \"Drop rows\":\n",
        "        data.dropna(inplace=True)\n",
        "    else:\n",
        "        for column in data.columns:\n",
        "            if data[column].dtype == np.number:\n",
        "                data[column].fillna(data[column].mean(), inplace=True)\n",
        "            else:\n",
        "                data[column].fillna(data[column].mode()[0], inplace=True)\n",
        "\n",
        "    # Encoding categorical data\n",
        "    categorical_columns = [col for col in all_columns if data[col].dtype == 'object']\n",
        "    if categorical_columns:\n",
        "        if st.checkbox(\"Encode categorical data?\"):\n",
        "            encoding_method = st.radio(\"Select encoding method\", (\"One-Hot Encoding\", \"Label Encoding\"))\n",
        "            if encoding_method == \"One-Hot Encoding\":\n",
        "                data = pd.get_dummies(data, columns=categorical_columns)\n",
        "            elif encoding_method == \"Label Encoding\":\n",
        "                from sklearn.preprocessing import LabelEncoder\n",
        "                le = LabelEncoder()\n",
        "                for col in categorical_columns:\n",
        "                    data[col] = le.fit_transform(data[col])\n",
        "\n",
        "    # Select X and Y\n",
        "    x = st.multiselect(\"Select features for X\", data.columns.tolist())\n",
        "    y = st.selectbox(\"Select target (Y)\", data.columns.tolist())\n",
        "\n",
        "    # Determine task type\n",
        "    if data[y].dtype in [np.float64, np.int64]:\n",
        "        task_type = \"Regression\"\n",
        "        setup = pc_regr.setup(data, target=y, silent=True)\n",
        "    else:\n",
        "        task_type = \"Classification\"\n",
        "        setup = pc_class.setup(data, target=y, silent=True)\n",
        "\n",
        "    # Train models with PyCaret\n",
        "    best_models = pc_regr.compare_models() if task_type == \"Regression\" else pc_class.compare_models()\n",
        "\n",
        "    # Display best models\n",
        "    st.write(\"Best Models:\", best_models)\n",
        "\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "p9MLlIjfhQqu",
        "outputId": "7f58d4b7-28cb-407d-ddda-b5c6c1af4701"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing streamlit_app.py\n"
          ]
        }
      ]
    }
  ]
}