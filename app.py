import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report

def main():
    # Page layout
    st.set_page_config(page_title="EDA Tool",
    initial_sidebar_state="expanded")
    st.set_option("deprecation.showPyplotGlobalUse", False)

    # Main title
    st.header("EDA")
    st.title("Exploratory Data Analysis (EDA) Tool")
    st.sidebar.title("Exploratory Data Analysis (EDA) Tool 📈")
    st.markdown("### By [Ansh Kapoor](https://github.com/AnshKapoor)")
    st.sidebar.markdown("By [Richard Cornelius Suwandi](https://github.com/richardcsuwandi)")
    st.sidebar.markdown("[![View on GitHub](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=GitHub)](https://github.com/richardcsuwandi/eda-tool)")
    st.sidebar.markdown("## Choose a theme")
    # App description
    st.markdown("### An exploratory analysis tool that provides various summaries and visualizations on the uploaded data.")

    # Upload file
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type="csv")

    st.info("Upload a CSV file to get started")
    if uploaded_file is not None:
        st.success("File sucessfully uploaded!")
        df = pd.read_csv(uploaded_file)
        numerical_col = df.select_dtypes(include=np.number).columns.tolist()
        categorical_col = df.drop(numerical_col, axis=1).columns

        # Show raw data
        if st.sidebar.checkbox("Show raw data", False):
            st.write(df)

        activity_list = ["Basic Exploratory Analysis", "Data Visualizations", "Pandas Profiling"]
        activity = st.sidebar.selectbox("Choose activity", activity_list)

        if activity == "Basic Exploratory Analysis":
            # Show head
            if st.sidebar.checkbox("Head", key="head"):
                st.subheader("DataFrame's Head")
                st.write(df.head())

            # Show tail
            if st.sidebar.checkbox("Tail", key="tail"):
                st.subheader("DataFrame's Tail")
                st.write(df.tail())

            # Show description
            if st.sidebar.checkbox("Describe", key="desc"):
                st.subheader("DataFrame's Description")
                st.write(df.describe())

            # Show missing values
            if st.sidebar.checkbox("Missing Values", key="mv"):
                st.subheader("DataFrame's Missing Values")
                st.write(df.isnull().sum())

            # Show unique values
            if st.sidebar.checkbox("Unique Values", key="unique"):
                col = st.sidebar.selectbox("Choose a column", df.columns)
                st.subheader(f"{col}'s Unique Values")
                st.write(df[col].value_counts())
            
            # Handle Missing Values
            if st.sidebar.checkbox("Handle Missing Values"):
                method = st.sidebar.radio("Fill or Drop?", ["Fill", "Drop"])
                if method == "Fill":
                    detailed_output = []
                    fill_value = st.sidebar.text_input("Fill value or method (mean/median)", "mean")
                    if fill_value in ["mean", "median"]:
                        fill_values = df.mean() if fill_value == "mean" else df.median()
                        for col in df.columns:
                            if col in fill_values:
                                missing_indices = df[df[col].isna()].index
                                for i in missing_indices:
                                    detailed_output.append(f"The missing value at index {i} in column '{col}' was replaced with {fill_values[col]:.2f}.")
                                df[col] = df[col].fillna(fill_values[col])
                        st.write(df)
                        
                    else:
                        for col in df.columns:
                            missing_indices = df[df[col].isna()].index
                            for i in missing_indices:
                                detailed_output.append(f"The missing value at index {i} in column '{col}' was replaced with '{fill_value}'.")
                            df = df.fillna(fill_value)
                            st.write(df)
                    for message in detailed_output:
                        st.text(message)

                else:
                    missing_indices = df[df.isna().any(axis=1)].index
                    detailed_output = []
                    for i in missing_indices:
                        detailed_output.append(f"Row at index {i} with missing values was dropped.")
                    df = df.dropna()
                    st.write(df)
                    for message in detailed_output:
                        st.text(message)
                st.write(df.isnull().sum())

        elif activity == "Data Visualizations":
            # Relation plot
            if st.sidebar.checkbox("Relational Plot", key="rel"):
                st.subheader("Relational Plot")
                if len(numerical_col) > 1:
                    x = st.sidebar.selectbox("Choose a column", numerical_col)
                    del numerical_col[numerical_col.index(x)]
                    y = st.sidebar.selectbox("Choose another column", numerical_col)
                    kind = st.sidebar.radio("Kind", ["scatter", "line"])
                    hue = st.sidebar.selectbox("Hue (Optional)", categorical_col.insert(0, None))
                    sns.relplot(x=x, y=y, data=df, kind=kind, hue=hue)
                    st.pyplot()
                else:
                    st.warning("Not enough columns to create plot")

            # Categorical plot
            if st.sidebar.checkbox("Categorical Plot", key="cat"):
                if (len(numerical_col) and len(categorical_col)) > 1:
                    x = st.sidebar.selectbox("Choose a column", categorical_col, key="cat_1")
                    y = st.sidebar.selectbox("Choose another column", numerical_col, key="cat_2")
                    kind_list = ["strip", "swarm", "box", "violin", "boxen", "point", "bar"]
                    kind = st.sidebar.selectbox("Kind", kind_list, key="cat_3")
                    st.subheader(f"{kind.capitalize()} Plot")
                    hue = st.sidebar.selectbox("Hue (Optional)", categorical_col.insert(0, None), key="cat_4")
                    sns.catplot(x=x, y=y, data=df, kind=kind, hue=hue)
                    st.pyplot()
                else:
                    st.warning("Not enough columns to create plot")

            # Count plot
            if st.sidebar.checkbox("Count Plot", False, key="count"):
                if len(categorical_col) > 1:
                    col = st.sidebar.selectbox("Choose a column", categorical_col, key="count")
                    st.subheader(f"{col}'s Count Plot'")
                    sns.countplot(x=col, data=df)
                    st.pyplot()
                else:
                    st.warning("Not enough columns to create plot")

            # Distribution plot
            if st.sidebar.checkbox("Distribution Plot", False, key="dist"):
                if len(numerical_col) > 1:
                    col = st.sidebar.selectbox("Choose a column", numerical_col,  key="dist")
                    st.subheader(f"{col}'s Distribution Plot")
                    sns.distplot(df[col])
                    plt.grid(True)
                    st.pyplot()
                else:
                    st.warning("Not enough columns to create plot")

            # Heatmap
            if st.sidebar.checkbox("Correlation Heatmap", False, key="heatmap"):
                st.subheader(f"Correlation Heatmap")
                sns.heatmap(df.corr(), annot=True)
                st.pyplot()

            # Pairplot
            if st.sidebar.checkbox("Pairplot", False, key="pairplot"):
                st.subheader(f"Pairplot")
                hue = st.sidebar.selectbox("Hue (Optional)", categorical_col.insert(0, None))
                sns.pairplot(df, hue=hue)
                st.pyplot()

if __name__ == "__main__":
    main()
