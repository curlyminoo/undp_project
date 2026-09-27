import streamlit as st
import pandas as pd
import plotly.express as px
from groq import Groq
import requests
import os


BACKEND_URL = "https://undp-backend-lr9n.onrender.com"


def safe_json_df(df):
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


st.title("🤖 AI Data Analyst")

upload_file = st.file_uploader(
    "📤 Upload your dataset",
    type=["csv", "xlsx", "json"]
)

if upload_file is not None:

    if st.session_state.get("uploaded_file_name") != upload_file.name:

        files = {"file": (upload_file.name, upload_file.getvalue())}

        with st.spinner("Uploading dataset..."):
            response = requests.post(f"{BACKEND_URL}/upload", files=files)

        if response.status_code == 200:
            data = response.json()

            if "error" in data:
                st.error(data["error"])
                st.stop()

            st.session_state["df"] = pd.DataFrame(data["data"])
            st.session_state["total_rows"] = data["rows"]
            st.session_state["total_columns"] = data["columns"]
            st.session_state["uploaded_file_name"] = upload_file.name
            st.session_state["model_trained"] = False
            st.session_state.pop("feature_cols", None)

            st.success("Dataset uploaded successfully.")
        else:
            st.error(f"Failed to upload dataset. Backend status: {response.status_code}")
            st.stop()

    df = st.session_state["df"]
    categorical_columns = df.select_dtypes(include="object").columns.tolist()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview & Cleaning",
        "🧹 Data Conversion",
        "📈 Charts & Insights",
        "🤖 ML Prediction"
    ])

    # ============ TAB 1: OVERVIEW & CLEANING ============
    with tab1:
        st.success("Dataset is ready for analysis.")
        st.write("Data preview:")

        if st.checkbox("Show dataset preview"):
            st.dataframe(df.head(100), use_container_width=True)

        st.subheader("📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Duplicate Rows", df.duplicated().sum())

        st.subheader("🧹 Data Cleaning")

        drop_dup = st.checkbox("Drop duplicate rows")
        fill_missing = st.checkbox("Fill missing values")

        if drop_dup or fill_missing:
            if st.button("Apply Cleaning"):
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

        st.subheader("✅ Data After Cleaning")
        if st.checkbox("Show the data frame"):
            st.dataframe(df)

    # ============ TAB 2: DATA CONVERSION ============
    with tab2:
        object_columns = df.select_dtypes(include='object').columns.tolist()

        if object_columns:
            st.subheader("🧹 Data Type Conversion")
            selected_col = st.multiselect("Select columns to convert to numeric:", object_columns)

            if st.button("🔄 Reset to Original File"):
                st.session_state.pop("df", None)
                st.session_state.pop("uploaded_file_name", None)
                st.session_state.pop("model_trained", None)
                st.session_state.pop("feature_cols", None)
                st.rerun()

            if st.button("Convert Selected Columns"):

                if not selected_col:
                    st.warning("Please select at least one column.")
                else:

                    payload = {
                        "data": safe_json_df(df),
                        "columns": selected_col
                    }

                    response = requests.post(f"{BACKEND_URL}/convert", json=payload)

                    if response.status_code == 200:
                        result = response.json()
                        df = pd.DataFrame(result["data"])
                        st.session_state["df"] = df

                        for warning in result["warnings"]:
                            st.warning(warning)

                        st.success(
                            f"{len(selected_col)} column(s) converted to numeric")

                        st.write(df[selected_col].head(10))

                    else:
                        st.error("Conversion failed.")

        else:
            st.info("No text columns available for conversion.")

    # ============ TAB 3: CHARTS & INSIGHTS ============
    with tab3:
        st.subheader("📈 Data Analysis and Charts")

        categorical_columns = df.select_dtypes(include="object").columns.tolist()
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        if categorical_columns:
            st.markdown("### 1️⃣ Category Distribution")
            count_col = st.selectbox("Select a categorical column:", categorical_columns, key="count_col")

            counts = df[count_col].value_counts().reset_index()
            counts.columns = [count_col, "Count"]

            fig1 = px.bar(counts, x=count_col, y="Count", title=f"Count by {count_col}")
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("No categorical columns found in this dataset.")

        if categorical_columns and numeric_columns:
            st.markdown("### 2️⃣ Average Numeric Value by Category")

            col1, col2 = st.columns(2)
            with col1:
                group_col = st.selectbox("Group by:", categorical_columns, key="group_col")
            with col2:
                value_col = st.selectbox("Calculate average of:", numeric_columns, key="value_col")

            avg_df = df.groupby(group_col)[value_col].mean().reset_index()
            fig2 = px.bar(avg_df, x=group_col, y=value_col, title=f"Average {value_col} by {group_col}")
            st.plotly_chart(fig2, use_container_width=True)
        elif numeric_columns:
            st.info("Numeric columns are available, but no categorical column was found for group comparison.")

        if len(numeric_columns) >= 2:
            st.markdown("### 3️⃣ Relationship Between Numeric Variables")

            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis:", numeric_columns, key="x_scatter")
            with col2:
                y_col = st.selectbox("Y-axis:", numeric_columns, index=1 if len(numeric_columns) > 1 else 0, key="y_scatter")

            if categorical_columns:
                color_col = st.selectbox("Color by (optional):", ["None"] + categorical_columns, key="color_scatter")
                if color_col == "None":
                    fig3 = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")
                else:
                    fig3 = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col} by {color_col}")
            else:
                fig3 = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}")

            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("At least two numeric columns are required to create a scatter plot.")

        if numeric_columns:
            st.markdown("### 4️⃣ Overall Numeric Statistics")
            statistics_df = df[numeric_columns].describe().T
            statistics_df = statistics_df[["count", "mean", "std", "min", "max"]].round(2)
            st.dataframe(statistics_df, use_container_width=True)
        else:
            st.info("No numeric columns found for statistical analysis.")

        st.subheader("💡 Key Insights")

        if numeric_columns:
            insight_numeric_col = st.selectbox("Select a numeric column:", numeric_columns, key="insight_numeric")

            mean_value = df[insight_numeric_col].mean()
            median_value = df[insight_numeric_col].median()
            min_value = df[insight_numeric_col].min()
            max_value = df[insight_numeric_col].max()

            st.write(f"• Average {insight_numeric_col}: {mean_value:.2f}")
            st.write(f"• Median {insight_numeric_col}: {median_value:.2f}")
            st.write(f"• Minimum {insight_numeric_col}: {min_value:.2f}")
            st.write(f"• Maximum {insight_numeric_col}: {max_value:.2f}")

            if categorical_columns:
                st.markdown("### Comparison by Category")

                insight_group_col = st.selectbox("Group by:", categorical_columns, key="insight_group")
                insight_value_col = st.selectbox("Analyze:", numeric_columns, key="insight_value")

                grouped = df.groupby(insight_group_col)[insight_value_col].agg(["mean", "min", "max"]).reset_index()
                grouped["mean"] = grouped["mean"].round(2)
                grouped["min"] = grouped["min"].round(2)
                grouped["max"] = grouped["max"].round(2)

                st.dataframe(grouped, use_container_width=True)
        else:
            st.info("No numeric columns are available for generating insights.")

    # ============ TAB 4: ML PREDICTION ============
    with tab4:
        st.subheader("🤖 ML Prediction")

        categorical_columns = df.select_dtypes(include="object").columns.tolist()
        numeric_columns = df.select_dtypes(include="number").columns.tolist()

        if categorical_columns and numeric_columns:
            target_col = st.selectbox("Select target column:", categorical_columns, key="target")
            feature_cols = st.multiselect("Select numeric feature columns:", numeric_columns, key="features")

            if target_col and feature_cols:
                if st.button("🚀 Train Model"):
                    train_payload = {
                        "data": safe_json_df(df),
                        "target_column": target_col,
                        "feature_columns": feature_cols
                    }

                    with st.spinner("Training model..."):
                        train_response = requests.post(f"{BACKEND_URL}/train", json=train_payload)

                    if train_response.status_code == 200:
                        result = train_response.json()

                        if "error" in result:
                            st.warning(result["error"])
                        else:
                            st.session_state["model_trained"] = True
                            st.session_state["feature_cols"] = feature_cols
                            st.session_state["target_col"] = target_col

                            accuracy = result.get("accuracy")
                            if accuracy is not None:
                                st.success(f"Model trained successfully. Accuracy: {accuracy:.2%}")
                            else:
                                st.success("Model trained successfully.")
                    else:
                        st.error(f"Training failed. Backend status: {train_response.status_code}")
        else:
            st.info("ML prediction requires at least one categorical column and one numeric column.")

        if st.session_state.get("model_trained"):
            st.subheader("🔮 Predict New Record")

            feature_cols = st.session_state["feature_cols"]
            input_data = {}

            for col in feature_cols:
                default_value = float(df[col].median())
                input_data[col] = st.number_input(f"Enter {col}:", value=default_value)

            if st.button("Predict"):
                predict_payload = {"input_data": input_data}

                with st.spinner("Generating prediction..."):
                    predict_response = requests.post(f"{BACKEND_URL}/predict", json=predict_payload)

                if predict_response.status_code == 200:
                    result = predict_response.json()

                    if "error" in result:
                        st.warning(result["error"])
                    else:
                        predicted_label = result.get("predicted_label")
                        confidence = result.get("confidence")

                        st.write(f"### 🎯 Prediction: {predicted_label}")
                        if confidence is not None:
                            st.write(f"Confidence: {confidence:.2%}")

                        groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")

                        if not groq_key:
                            st.warning("GROQ_API_KEY is not configured.")
                        else:
                            client = Groq(api_key=groq_key, timeout=30.0)

                            confidence_text = f"{confidence:.2%}" if confidence is not None else "Not available"

                            prompt = f"""
                                        A dataset was analyzed.

                                        Input features:
                                        {input_data}

                                        The machine learning model predicted:
                                        {predicted_label}

                                        Model confidence:
                                        {confidence_text}

                                        Explain this result in simple language.

                                        Use only the information provided above.
                                        Do not invent facts.
                                        Do not claim that a feature caused the prediction.
                                        Do not assume this is a development project.

                                        Keep the explanation under 100 words.
                                        """

                            with st.spinner("Generating AI explanation..."):
                                ai_response = client.chat.completions.create(
                                    model="openai/gpt-oss-20b",
                                    messages=[{"role": "user", "content": prompt}]
                                )

                            st.write("### 🧠 AI Explanation")
                            st.write(ai_response.choices[0].message.content)
                else:
                    st.error("Prediction failed.")