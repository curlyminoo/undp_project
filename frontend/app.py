import streamlit as st
import pandas as pd
import openpyxl
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn .preprocessing import LabelEncoder
from groq import Groq
import requests
import os


BACKEND_URL = "https://undp-backend-lr9n.onrender.com"


def safe_json_df(df):
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")



# upload file
st.title("Developement UNDP Project")

upload_file = st.file_uploader("Uploade your file" ,type= ["csv", "xlsx", "json"])

if upload_file is not None:
    if 'df' not in st.session_state:

        files = {"file" : (upload_file.name, upload_file.getvalue())}
        response = requests.post(f"{BACKEND_URL}/upload", files=files)

        if response.status_code == 200:
            data = response.json()

            if "error" in data:
                st.error(data["error"])
                st.stop()

            else:
                st.session_state['df'] = pd.DataFrame(data["data"])
                st.session_state['total_rows'] = data["rows"]
                st.session_state['total_columns'] = data["columns"]

        else:
            st.error("Failed to connect to the backend server.")
            st.stop()

    df = st.session_state['df']

    st.success("upload successfully")
    st.write("data frame: ")
    st.dataframe(df)


    # overview
    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        st.metric("Duplicate Rows", df.duplicated().sum())
    

    # data cleaning
    st.subheader("Data Cleaning")

    drop_dup = st.checkbox("Drop duplicate Rows: ")
    fill_missing = st.checkbox("fill the missing values: ")


    if drop_dup or fill_missing:
        if st.button("Apply Cleaning"):
            df_clean_for_json = df.astype(object).where(df.notna(), None)

            payload = {
                "data": safe_json_df(df),
                "drop_duplicates": drop_dup,
                "fill_missing": fill_missing
                }

            response = requests.post(f"{BACKEND_URL}/clean", json=payload)

            if response.status_code == 200:
                result = response.json()
                df = pd.DataFrame(result["data"])
                st.session_state['df'] = df
                st.success(f"Rows before: {result['rows_before']}, after: {result['rows_after']}")
            else:
                st.error("Cleaning failed.")


    st.subheader("data after cleaning: ")
    if st.checkbox("show the data frame"):
        st.dataframe(df)



    # datatype conversion
    object_columns = df.select_dtypes(include='object').columns.tolist()

    if object_columns:
        st.subheader("Data type Conversion")
        selected_col = st.multiselect("Select the columns to convert to numeric: ", object_columns)

        if st.button("Reset to original file"):
            del st.session_state['df']
            st.rerun()
        if st.button("Convert selected columns"):
            payload = {
                "data": safe_json_df(df),
                "columns": selected_col
            }
            response = requests.post(f"{BACKEND_URL}/convert", json=payload)

            if response.status_code == 200:
                result = response.json()
                df = pd.DataFrame(result["data"])
                st.session_state['df'] = df

                for warning in result["warnings"]:
                    st.warning(warning)

                st.success(f"{len(selected_col)} column(s) converted to numeric")
                st.write(df[selected_col].head(10))
            else:
                st.error("Conversion failed.")

        
        

    # charts
    st.subheader("Data Analysis and Charts")

    categorical_columns = df.select_dtypes(include="object").columns.to_list()
    numeric_columns = df.select_dtypes(include="number").columns.to_list()

    # نمودار اول: Count by category
    if categorical_columns:
        count_col = st.selectbox("select a category column to count: ", categorical_columns, key="count_col")

        payload = {"data": safe_json_df(df), "count_column": count_col}
        response = requests.post(f"{BACKEND_URL}/analyze", json=payload)

        if response.status_code == 200:
            analysis = response.json()
            if analysis["counts_by_category"]:
                cat_data = analysis["counts_by_category"]
                counts_df = pd.DataFrame(cat_data["data"])
                fig1 = px.bar(counts_df, x=cat_data["column"], y='Count',
                        title=f"Count by {cat_data['column']}")
                st.plotly_chart(fig1)
        else:
            st.error("Analysis failed.")

    # نمودار دوم: Average by group
    if categorical_columns and numeric_columns:
        group_col = st.selectbox("Group by: ", categorical_columns, key="group_col")
        value_col = st.selectbox("Average of: ", numeric_columns, key="value_col")

        payload2 = {"data": safe_json_df(df), "group_column": group_col, "value_column": value_col}
        response2 = requests.post(f"{BACKEND_URL}/analyze", json=payload2)

        if response2.status_code == 200:
            analysis2 = response2.json()
            if analysis2["averages_by_group"]:
                avg_data = analysis2["averages_by_group"]
                avg_df = pd.DataFrame(avg_data["data"])
                fig2 = px.bar(avg_df, x=avg_data["group_column"], y=avg_data["value_column"],
                        title=f"Average {avg_data['value_column']} by {avg_data['group_column']}")
                st.plotly_chart(fig2)
        else:
            st.error("Analysis failed.")

    # نمودار سوم: Scatter
    if len(numeric_columns) >= 2 and categorical_columns:
        st.write("Relationship between two numeric variables")

        x_col = st.selectbox("X-axis:", numeric_columns, key="x_scatter")
        y_col = st.selectbox("Y-axis:", numeric_columns, key="y_scatter")
        color_col = st.selectbox("Color by:", categorical_columns, key="color_scatter")

        scatter_payload = {
            "data": safe_json_df(df),
            "x_column": x_col,
            "y_column": y_col,
            "color_column": color_col
        }
        scatter_response = requests.post(f"{BACKEND_URL}/scatter", json=scatter_payload)

        if scatter_response.status_code == 200:
            scatter_data = scatter_response.json()
            scatter_df = pd.DataFrame(scatter_data["data"])
            fig3 = px.scatter(scatter_df, x=x_col, y=y_col, color=color_col,
                    title=f"{y_col} vs {x_col} by {color_col}")
            st.plotly_chart(fig3)
        else:
            st.error("Scatter analysis failed.")


    # key insight
    if categorical_columns and numeric_columns:
        st.subheader("Key Insights")

        unique_value = df[categorical_columns[0]].unique()
        preview_count = 10

        show_all = st.checkbox(f"Show all {len(unique_value)} categories") if len(unique_value) > preview_count else True
        values_to_show = unique_value if show_all else unique_value[:preview_count]

        for status_val in values_to_show:
            avg_val = df[df[categorical_columns[0]] == status_val][numeric_columns[0]].mean()
            st.write(f"- Average {numeric_columns[0]} for '{status_val}': {avg_val:.2f}")

    elif numeric_columns:
        st.info("No categorical column found to group by. Showing overall stats instead:")
        for col in numeric_columns:
            st.write(f"- Average {col}: {df[col].mean():.2f}")
    else:
        st.info("Not enough data to generate insights.")


    # ML model
    st.subheader("Risk Prediction Model")

    if categorical_columns and numeric_columns:
        target_col = st.selectbox("Select target column (what to predict): ", categorical_columns, key='target')
        feature_cols = st.multiselect("Select feature columns (inputs): ", numeric_columns, key='features')

        if target_col and feature_cols:
            if st.button("Train model"):
                train_payload = {
                    "data": safe_json_df(df),
                    "target_column": target_col,
                    "feature_columns": feature_cols
                }
                train_response = requests.post(f"{BACKEND_URL}/train", json=train_payload)

                if train_response.status_code == 200:
                    result = train_response.json()
                    if "error" in result:
                        st.warning(result["error"])
                    else:
                        st.session_state['model_trained'] = True
                        st.session_state['feature_cols'] = feature_cols
                        st.success(f"Model trained! Accuracy: {result['accuracy']:.2%}")
                else:
                    st.error("Training failed.")

    # predict new project
    if st.session_state.get('model_trained'):
        st.subheader("Predict New Project")

        feature_cols = st.session_state['feature_cols']

        input_data = {}
        for col in feature_cols:
            input_data[col] = st.number_input(f"Enter {col}: ", value=float(df[col].median()))

        if st.button("Predict"):
            predict_payload = {"input_data": input_data}
            predict_response = requests.post(f"{BACKEND_URL}/predict", json=predict_payload)

            if predict_response.status_code == 200:
                result = predict_response.json()
                if "error" in result:
                    st.warning(result["error"])
                else:
                    predicted_label = result["predicted_label"]
                    confidence = result["confidence"]

                    st.write(f"### Predicted: {predicted_label}")
                    st.write(f"Confidence: {confidence:.2%}")

                    # AI Explanation
                    groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
                    client = Groq(api_key=groq_key, timeout=30.0)
                    available_models = client.models.list()
                    model_names = [m.id for m in available_models.data]
                    preferred_models = ["openai/gpt-oss-20b", "llama-3.1-8b-instant", "openai/gpt-oss-120b"]
                    selected_model = next((m for m in preferred_models if m in model_names), model_names[0])

                    prompt = f"""A development project was analyzed with these characteristics: {input_data}

                    The model predicted: {predicted_label} with {confidence:.2%} confidence.

                    Explain in simple, non-technical language why this project might have this risk level,
                    based on general patterns in project management (budget size, duration, etc).
                    Keep it under 100 words."""

                    with st.spinner("Generating explanation..."):
                        ai_response = client.chat.completions.create(
                            model=selected_model,
                            messages=[{"role": "user", "content": prompt}]
                        )

                    st.write("### AI Explanation")
                    st.write(ai_response.choices[0].message.content)
            else:
                st.error("Prediction failed.")


