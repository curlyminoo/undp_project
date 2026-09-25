# UNDP Development Project Risk Analyzer

A web application that helps analyze development project data, identify patterns, predict delay risks, and generate simple, human-readable explanations using AI.

## 🔗 Live Demo
- **App:** https://undp-streamlit.onrender.com
- **API Docs:** https://undp-backend-lr9n.onrender.com

> Note: The backend may take 30-60 seconds to wake up on first use (free hosting tier).

## What it does

1. **Upload** a CSV, Excel, or JSON file with project data
2. **Clean** the data (remove duplicates, handle missing values)
3. **Convert** text columns to numeric where needed
4. **Visualize** patterns with interactive charts
5. **Train** a machine learning model to predict project delay risk
6. **Get AI explanations** in plain language for any prediction

## Tech Stack

- **Backend:** FastAPI, Pandas, Scikit-learn
- **Frontend:** Streamlit, Plotly
- **AI:** Groq API (Llama/GPT-OSS models)
- **Deployment:** Render

## Architecture