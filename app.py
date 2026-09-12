import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Scania APS Predictive Maintenance", page_icon="🚛")
st.title("🚛 Truck Fleet Maintenance Predictor")
st.write("Upload a batch of raw truck sensor telemetry to identify imminent Air Pressure System (APS) failures.")


# 2. Load the Production Pipeline
# The @st.cache_resource decorator keeps the model in memory so it doesn't reload on every click
@st.cache_resource
def load_pipeline():
    return joblib.load('models/scania_lgbm_pipeline.pkl')


pipeline = load_pipeline()

# 3. Create a File Uploader in the UI
uploaded_file = st.file_uploader("Upload Truck Sensor Data (CSV)", type="csv")

# 4. Process the Data if a file is uploaded
if uploaded_file is not None:
    st.info("File uploaded successfully. Processing data...")

    # Read the CSV
    df = pd.read_csv(uploaded_file, na_values="na")

    # Ensure the class column is dropped if it exists in the uploaded file
    if 'class' in df.columns:
        X_live = df.drop(columns=['class'])
    else:
        X_live = df

    # Align the columns to match what the imputer expects
    X_aligned = X_live[pipeline['features']]

    # Make Predictions using the pipeline's components
    probabilities = pipeline['model'].predict_proba(X_aligned)[:, 1]
    predictions = (probabilities >= pipeline['optimal_threshold']).astype(int)

    # 5. Display the Results
    total_trucks = len(predictions)
    failures_detected = sum(predictions)

    st.success("Analysis Complete!")

    # Show high-level metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Trucks Analyzed", total_trucks)
    col2.metric("⚠️ Imminent Failures Detected", failures_detected)

    if failures_detected > 0:
        st.warning(
            f"Action Required: {failures_detected} trucks require immediate mechanic inspection to prevent breakdown.")
    else:
        st.balloons()
        st.success("All trucks are operating within healthy parameters!")