# streamlit_app.py
import os
import requests
import pandas as pd
import streamlit as st
from typing import Dict, Any

# ------------------------
# Config / user dataset
# ------------------------
DATASET_PATH = "data/diabetes_dataset.csv"
# url = "https://raw.githubusercontent.com/kaisarhossain/Diabetes-Risk-Prediction/refs/heads/main/diabetes_dataset.csv"
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000/predict")
HF_MODEL_URL = "https://api-inference.huggingface.co/models/kaisarhossain/diabetes-risk-pred-kaisar-v1.1"
HF_API_TOKEN = os.getenv("HF_TOKEN")  # set in your environment if using HF


# ------------------------
# Helper functions
# ------------------------
def call_fastapi(payload: Dict[str, Any]):
    try:
        resp = requests.post(FASTAPI_URL, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def call_hf_model(payload: Dict[str, Any]):
    if not HF_API_TOKEN:
        return {"error": "HF_API_TOKEN not set in environment"}
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    try:
        r = requests.post(HF_MODEL_URL, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ------------------------
# Streamlit UI
# ------------------------

## Setting ST background
# import base64
#
# @st.cache_data
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, "rb") as f:
#         data = f.read()
#     return base64.b64encode(data).decode()
#
#
# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = f"""
#         <style>
#         .stApp {{
#             background-image: url("data:image/png;base64,{bin_str}");
#             background-size: cover;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#         }}
#         </style>
#     """
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#
#
# # Call the function
# set_png_as_page_bg("background.png")
## Setting ST background

st.set_page_config(page_title="Diabetes Prediction UI", layout="wide", initial_sidebar_state="expanded")

# Sidebar nav
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Preview", "EDA Quick View", "Predict (Single)", "Batch Predict"])

st.sidebar.markdown("---")
st.sidebar.write("Model endpoints:")
st.sidebar.write(f"- FastAPI: `{FASTAPI_URL}`")
st.sidebar.write(f"- Hugging Face: `{HF_MODEL_URL.split('/')[-1]}`")
st.sidebar.markdown("---")
st.sidebar.caption("Created by: Kaisar Hossain")


# ------------------------
# Shared UI: input form factory
# ------------------------
def input_form(prefix=""):
    a1c = st.number_input(f"{prefix}HbA1c (%)", min_value=3.0, max_value=20.0, value=5.6, step=0.1)
    glucose_pp = st.number_input(f"{prefix}Postprandial Glucose (mg/dL)", min_value=40, max_value=500, value=120,
                                 step=1)
    glucose_f = st.number_input(f"{prefix}Fasting Glucose (mg/dL)", min_value=40, max_value=400, value=95, step=1)
    family_hist = st.selectbox(f"{prefix}Family History", ["No", "Yes"])
    age = st.number_input(f"{prefix}Age (yrs)", 1, 120, 45, step=1)
    activity = st.number_input(f"{prefix}Physical Activity (min/week)", 0, 2000, 40, step=5)
    bmi = st.number_input(f"{prefix}BMI", min_value=10.0, max_value=70.0, value=27.5, step=0.1)
    systolic = st.number_input(f"{prefix}Systolic BP (mmHg)", 60, 250, 130, step=1)
    return dict(
        a1c=[round(a1c, 2)],
        glucose_pp=[int(glucose_pp)],
        glucose_fast=[int(glucose_f)],
        family_history=[family_hist],
        age=[int(age)],
        activity=[int(activity)],
        bmi=[round(bmi, 2)],
        systolic_bp=[int(systolic)]
    )


# ------------------------
# Page: Home
# ------------------------
if page == "Home":
    st.title("🩺 Diabetes Risk Prediction — Dashboard")
    st.markdown(
        "This dashboard enables to predict diabetic risk based on eight major parameters. "
        "You can use a local FastAPI model (recommended) or call my Hugging Face hosted model for the risk prediction."
    )
    st.markdown("---")
    st.subheader("How to use")
    st.markdown("""
    1. Start the FastAPI server runs on port 8000 (optional): `uvicorn fastapi_server:app --reload`  
    2. Use Hugging Face inference and uploaded model.  
    3. Go to **Predict (Single)** for per-patient predictions, or **Batch Predict** to upload CSV for bulk scoring.
    """)

# ------------------------
# Page: Data Preview
# ------------------------

elif page == "Data Preview":
    st.title("Dataset Preview")
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH, nrows=10000)  # load first chunk
        st.dataframe(df.head(100))
        st.markdown(f"Full dataset path: `{DATASET_PATH}`")
        st.metric("Rows (sample shown)", len(df))
    else:
        st.error(f"Dataset not found at {DATASET_PATH}")

# ------------------------
# Page: EDA Quick View
# ------------------------
elif page == "EDA Quick View":
    st.title("Quick EDA")
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)

        st.subheader("Target Balance")
        if "diagnosed_diabetes" in df.columns:
            counts = df["diagnosed_diabetes"].value_counts(normalize=True)
            st.bar_chart(counts)

        st.subheader("HbA1c distribution")
        if "hba1c" in df.columns:
            st.write(df["hba1c"].describe())

            import altair as alt

            hist = alt.Chart(df).mark_bar().encode(
                x=alt.X("hba1c:Q", bin=alt.Bin(step=0.2)),
                y='count()'
            ).properties(width=700)

            st.altair_chart(hist, use_container_width=True)

        st.subheader("Sample Data")
        st.data_editor(df.head(200))
    else:
        st.error("Dataset not available")


# ------------------------
# Page: Predict (Single)
# ------------------------
elif page == "Predict (Single)":
    st.title("Single Patient Prediction")
    st.markdown("Enter patient values and choose model source (local FastAPI or Hugging Face).")
    with st.form(key="predict_form"):
        model_source = st.radio("Model source", ("FastAPI (local)", "Hugging Face (remote)"))
        inputs = input_form(prefix="")
        submitted = st.form_submit_button("Predict")
    if submitted:
        st.info("Running prediction...")
        payload = inputs  # the model endpoints expect {a1c: [...], ...}
        result = None
        if model_source == "FastAPI (local)":
            result = call_fastapi(payload)
            if result and "error" in result:
                st.error("FastAPI error: " + result["error"])
                st.info("Try using Hugging Face option or check FastAPI server.")
                result = None
        if (not result) and model_source == "Hugging Face (remote)":
            result = call_hf_model({"inputs": payload})
            # HF could return nested or top-level structure; attempt to normalize
            if isinstance(result, dict) and "response" in result and isinstance(result["response"], dict):
                result = result["response"]
        if result:
            # Expecting {'preds':[...],'probs':[...]} or similar
            preds = result.get("preds") or result.get("predictions") or []
            probs = result.get("probs") or result.get("probabilities") or result.get("scores") or []
            if probs:
                prob = float(probs[0])
                pred = preds[0] if preds else ("Yes" if prob >= 0.5 else "No")
                st.success(f"Prediction: {pred} — {round(prob * 100, 1)}%")
                st.metric("Risk probability", f"{round(prob * 100, 1)}%")
            else:
                st.error("Model did not return probability. Raw response: " + str(result))

# ------------------------
# Page: Batch Predict
# ------------------------
elif page == "Batch Predict":
    st.title("Batch Predict (CSV upload)")
    st.markdown(
        "Upload CSV with columns: hba1c, glucose_postprandial, glucose_fasting, family_history, age, activity, bmi, systolic_bp")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.write("Rows:", len(df))
        # prepare payload in chunks and call FastAPI or HF
        use_fastapi = st.checkbox("Use FastAPI (local) if available", value=True)
        if st.button("Run batch predict"):
            # chunk to avoid request size limits
            BATCH = 2000
            results = []
            for i in range(0, len(df), BATCH):
                chunk = df.iloc[i:i + BATCH]
                payload = {
                    "a1c": chunk["hba1c"].tolist(),
                    "glucose_pp": chunk["glucose_postprandial"].tolist(),
                    "glucose_fast": chunk["glucose_fasting"].tolist(),
                    "family_history": chunk["family_history"].tolist(),
                    "age": chunk["age"].astype(int).tolist(),
                    "activity": chunk["activity"].astype(int).tolist(),
                    "bmi": chunk["bmi"].tolist(),
                    "systolic_bp": chunk["systolic_bp"].astype(int).tolist()
                }
                if use_fastapi:
                    resp = call_fastapi(payload)
                    if resp and "error" in resp:
                        st.error("FastAPI error: " + resp["error"])
                        break
                    preds = resp.get("preds", [])
                    probs = resp.get("probs", [])
                else:
                    resp = call_hf_model({"inputs": payload})
                    if isinstance(resp, dict) and "response" in resp and isinstance(resp["response"], dict):
                        resp = resp["response"]
                    preds = resp.get("preds", [])
                    probs = resp.get("probs", [])
                # append to chunk
                chunk_res = chunk.copy()
                chunk_res["predicted_class"] = preds[:len(chunk_res)]
                chunk_res["predicted_prob"] = probs[:len(chunk_res)]
                results.append(chunk_res)
            if results:
                out = pd.concat(results, ignore_index=True)
                st.success("Batch prediction completed")
                st.dataframe(out.head(200))
                st.download_button("Download scored CSV", out.to_csv(index=False).encode("utf-8"), "scored.csv")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.caption("Created by: Kaisar Hossain")
