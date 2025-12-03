import streamlit as st
import pandas as pd
import numpy as np
import joblib            # ← use joblib instead of pickle
from pathlib import Path

# --------------------------
# 1. Model loader
# --------------------------
@st.cache_resource
def load_pipeline(model_path: str = "xgboost_pipeline.pkl"):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Make sure it is in the same folder "
            f"as this streamlit_app.py file."
        )

    try:
        pipeline = joblib.load(path)   # ← this is the important change
    except Exception as e:
        # Show a nice error in Streamlit so you can see what's wrong
        st.error(f"Failed to load model with joblib: {e}")
        raise

    return pipeline

pipeline = load_pipeline()
