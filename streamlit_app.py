import streamlit as st
import pandas as pd
import os
import pickle
import time
from typing import Tuple
import constants as cons
from app.helper_functions import get_transformer

def load_model(model_path: str):
    """Load trained model from pickle file."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)

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

def get_model_path(experiment_id: str) -> Tuple[str, str]:
    """Get model and transformer paths for selected experiment"""
    if experiment_id == "best_model":
        base_path = "data/best_model"
    else:
        base_path = f"archived_experiments/experiment_{experiment_id}"
    
    model_path = os.path.join(base_path, "train", cons.DEFAULT_MODEL_FILE)
    transformer_path = os.path.join(base_path, "preprocess")
    
    return model_path, transformer_path

def main():
    st.title("Click Baiters")
    
    # File upload
    uploaded_file = st.file_uploader("Upload test data (CSV)", type="csv")
    
    # Model selection
    experiments = get_available_experiments()
    selected_experiment = st.selectbox(
        "Select experiment", 
        experiments,
        index=experiments.index("best_model") if "best_model" in experiments else 0
    )
    
    if uploaded_file and selected_experiment:
        # Load test data
        test_data = pd.read_csv(uploaded_file)
        
        # Get model and transformer paths
        model_path, transformer_path = get_model_path(selected_experiment)
        
        if st.button("Generate Predictions"):
            with st.spinner("Processing..."):
                # Create placeholder for progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load model and transformer
                status_text.text("Loading model...")
                progress_bar.progress(20)
                model = load_model(model_path)
                
                status_text.text("Loading transformer...")
                progress_bar.progress(40)
                transformer = get_transformer(transformer_path)
                
                # Preprocess data
                status_text.text("Preprocessing data...")
                progress_bar.progress(60)
                processed_data = transformer.transform(test_data)
                
                # Generate predictions
                status_text.text("Generating predictions...")
                progress_bar.progress(80)
                predictions = pd.DataFrame(
                    model.predict(processed_data),
                    columns=[cons.TARGET_COLUMN]
                )
                
                # # Save predictions temporarily
                # temp_path = f"temp_predictions_{int(time.time())}.csv"
                # predictions.to_csv(temp_path, index=False)
                
                progress_bar.progress(100)
                status_text.text("Done!")
                
                # Provide download button
                # with open(temp_path, 'rb') as f:
                #     st.download_button(
                #         label="Download Predictions",
                #         data=f,
                #         file_name="predictions.csv",
                #         mime="text/csv"
                #     )
                
                # # Cleanup temp file
                # os.remove(temp_path)

if __name__ == "__main__":
    main() 