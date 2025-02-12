import streamlit as st
import pandas as pd
import os
import subprocess
from typing import Tuple
from experiments import Experiment

def get_available_experiments() -> list:
    """Get list of available experiment run_ids"""
    experiments = []
    # Add best model as default option 
    if os.path.exists("data/best_model"):
        experiments.append("best_model")
    
    # Add archived experiments
    archive_path = "archived_experiments"
    if os.path.exists(archive_path):
        experiments.extend([
            d.replace("experiment_", "") 
            for d in os.listdir(archive_path) 
            if d.startswith("experiment_")
        ])
    
    return experiments

def run_inference_pipeline(selected_experiment: str, csv_for_prediction: str):
    try:
        subprocess.run([
            "invoke", "inference-pipeline",
            f"--run-id={selected_experiment}",
            f"--csv-for-prediction={csv_for_prediction}"
        ], check=True)
        st.success("Predictions generated successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Error generating predictions: {str(e)}")
    finally:
        if os.path.exists(csv_for_prediction):
            os.remove(csv_for_prediction)

def main():

    st.title("Click Baiters")
    
    uploaded_file = st.file_uploader("Upload CSV for prediction", type="csv")
    
    experiments = get_available_experiments()
    selected_experiment = st.selectbox(
        "Select experiment", 
        experiments,
        index=experiments.index("best_model") if "best_model" in experiments else 0
    )
    
    if uploaded_file and selected_experiment:

        temp_input_csv_for_prediction = f"streamlit_temp_input_{uploaded_file.name}"
        with open(temp_input_csv_for_prediction, "wb") as f:
            f.write(uploaded_file.getvalue())

        print(f"Temp input path: {temp_input_csv_for_prediction}")
            
        if st.button("Generate Predictions"):
            with st.spinner("Processing..."):

                run_inference_pipeline(selected_experiment, temp_input_csv_for_prediction)

                predictions_path = Experiment(run_id=selected_experiment).predictions_path

                if predictions_path and os.path.exists(predictions_path):
                    with open(predictions_path, "rb") as file:
                        st.download_button(
                            label="Download Predictions",
                            data=file,
                            file_name=os.path.basename(predictions_path),
                            mime="text/csv"
                        )

if __name__ == "__main__":
    main()