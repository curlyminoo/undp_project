from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


app = FastAPI()
model_store = {}


def safe_json(df):
    df = df.replace([float("inf"), float("-inf")], None)
    return df.astype(object).where(df.notna(), None)


# classes
class CleanRequest(BaseModel):
    data: list
    drop_duplicates: bool = False
    fill_missing: bool = False

class ConvertRequest(BaseModel):
    data: list
    columns: list

class AnalyzeRequest(BaseModel):
    data: list
    count_column: str = None
    group_column: str = None
    value_column: str = None

class ScatterRequest(BaseModel):
    data: list
    x_column: str
    y_column: str
    color_column: str

class TrainRequest(BaseModel):
    data: list
    target_column: str
    feature_columns: list

class PredictRequest(BaseModel):
    input_data: dict



# upload file
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    file_name = file.filename.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(contents))
    elif file_name.endswith(".json"):
        df = pd.read_json(io.BytesIO(contents))
    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(io.BytesIO(contents))
    else:
        return {"error": "File format not supported"}

    df = safe_json(df)

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.to_list(),
        "data": df.to_dict(orient="records")
    }

# cleaning data
@app.post("/clean")
async def clean_data(request : CleanRequest):
    df = pd.DataFrame(request.data)

    before = len(df)

    if request.drop_duplicates:
        df = df.drop_duplicates().reset_index(drop= True)

    if request.fill_missing:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("unknown")

    after = len(df)

    df = safe_json(df)

    return {
        "rows_before": before,
        "rows_after": after,
        "data": df.to_dict(orient="records")
    }


# convert
@app.post("/convert")
async def convert_columns(request: ConvertRequest):
    df = pd.DataFrame(request.data)

    warnings = []

    for col in request.columns:
        original_non_null = df[col].notna().sum()

        cleaned_col = df[col].astype(str)
        cleaned_col = cleaned_col.str.replace("$", "", regex=False)
        cleaned_col = cleaned_col.str.replace(",", "", regex=False)
        cleaned_col = cleaned_col.str.replace(" ", "", regex=False)
        cleaned_col = cleaned_col.str.replace(r'(\d)[kK]$', r'\g<1>000', regex=True)
        df[col] = pd.to_numeric(cleaned_col, errors='coerce')

        new_non_null = df[col].notna().sum()
        failed_count = original_non_null - new_non_null

        if failed_count > 10:
            warnings.append(f"'{col}': {failed_count} value(s) could not be converted.")

    df = safe_json(df)

    return {
        "warnings": warnings,
        "data": df.to_dict(orient="records")
    }


# analyze
@app.post("/analyze")
async def analyze_data(request: AnalyzeRequest):
    df = pd.DataFrame(request.data)

    categorical_columns = df.select_dtypes(include="object").columns.to_list()
    numeric_columns = df.select_dtypes(include="number").columns.to_list()

    result = {
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "counts_by_category": {},
        "averages_by_group": {}
    }

    count_col = request.count_column or (categorical_columns[0] if categorical_columns else None)
    if count_col:
        counts = df[count_col].value_counts().reset_index()
        counts.columns = [count_col, 'Count']
        counts = safe_json(counts)
        result["counts_by_category"] = {
            "column": count_col,
            "data": counts.to_dict(orient="records")
        }

    group_col = request.group_column or (categorical_columns[0] if categorical_columns else None)
    value_col = request.value_column or (numeric_columns[0] if numeric_columns else None)
    if group_col and value_col:
        avg_data = df.groupby(group_col)[value_col].mean().reset_index()
        avg_data = safe_json(avg_data)
        result["averages_by_group"] = {
            "group_column": group_col,
            "value_column": value_col,
            "data": avg_data.to_dict(orient="records")
        }

    return result


# scatter chart
@app.post("/scatter")
async def scatter_data(request: ScatterRequest):
    df = pd.DataFrame(request.data)

    scatter_df = df[[request.x_column, request.y_column, request.color_column]].dropna()
    scatter_df = safe_json(scatter_df)

    return {
        "data": scatter_df.to_dict(orient="records")
    }


# train model
@app.post("/train")
async def train_model(request: TrainRequest):
    df = pd.DataFrame(request.data)
    model_df = df[request.feature_columns + [request.target_column]].dropna()

    if len(model_df) < 10:
        return {"error": "Not enough data to train a reliable model (minimum 10 rows recommended)."}

    if model_df[request.target_column].nunique() < 2:
        return {"error": f"'{request.target_column}' has only one unique value — cannot train a model."}

    X = model_df[request.feature_columns]
    y = model_df[request.target_column]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    model_store["model"] = model
    model_store["label_encoder"] = le
    model_store["feature_columns"] = request.feature_columns

    return {"accuracy": accuracy}


# predict
@app.post("/predict")
async def predict(request: PredictRequest):
    if "model" not in model_store:
        return {"error": "No trained model found. Please train a model first."}

    model = model_store["model"]
    le = model_store["label_encoder"]

    input_df = pd.DataFrame([request.input_data])
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    predicted_label = le.inverse_transform(prediction)[0]
    confidence = float(prediction_proba.max())

    return {
        "predicted_label": predicted_label,
        "confidence": confidence
    }