import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
    with open(path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline

pipeline = load_pipeline()

# --------------------------
# 2. Static lists (must match training & publisher)
# --------------------------

# Device name -> type (copied from your streaming script)
DEVICE_MAPPING = {
    "Alaris GH": "Infusion Pump",
    "Baxter AK 96": "Dialysis Machine",
    "Baxter Flo-Gard": "Infusion Pump",
    "Datex Ohmeda S5": "Anesthesia Machine",
    "Drager Fabius Trio": "Anesthesia Machine",
    "Drager V500": "Patient Ventilator",
    "Fresenius 4008": "Dialysis Machine",
    "GE Aisys": "Anesthesia Machine",
    "GE Logiq E9": "Ultrasound Machine",
    "GE MAC 2000": "ECG Monitor",
    "GE Revolution": "CT Scanner",
    "Hamilton G5": "Patient Ventilator",
    "HeartStart FRx": "Defibrillator",
    "Lifepak 20": "Defibrillator",
    "NxStage System One": "Dialysis Machine",
    "Philips EPIQ": "Ultrasound Machine",
    "Philips HeartStrart": "Defibrillator",
    "Philips Ingenuity": "CT Scanner",
    "Phillips PageWriter": "ECG Monitor",
    "Puritan Bennett 980": "Patient Ventilator",
    "Siemens Acuson": "Ultrasound Machine",
    "Siemens S2000": "Ultrasound Machine",
    "Smiths Medfusion": "Infusion Pump",
    "Zoll R Series": "Defibrillator",
}

DEVICE_NAMES = list(DEVICE_MAPPING.keys())

# Hospitals & regions – keep in sync with Location_list used in training
HOSPITALS = [
    "Hospital A", "Hospital B", "Hospital C",
    "Hospital D", "Hospital E", "Hospital F",
    "Hospital G", "Hospital H",
]

REGIONS = [
    "Central Region",
    "East Region",
    "North Region",
    "South Region",
    "West Region",
]

LOCATION_OPTIONS = [f"{h} - {r}" for h in HOSPITALS for r in REGIONS]

CLIMATE_OPTIONS = ["Yes", "No"]

# These are the feature columns expected by the pipeline
FEATURE_COLUMNS = [
    "DeviceType",
    "DeviceName",
    "RuntimeHours",
    "TemperatureC",
    "PressureKPa",
    "VibrationMM_S",
    "CurrentDrawA",
    "SignalNoiseLevel",
    "ClimateControl",
    "HumidityPercent",
    "Location",
    "OperationalCycles",
    "UserInteractionsPerDay",
    "ApproxDeviceAgeYears",
    "NumRepairs",
    "ErrorLogsCount",
]


# --------------------------
# 3. Streamlit UI
# --------------------------
st.set_page_config(
    page_title="Medical Device Failure Risk – Streamlit Dashboard",
    layout="wide",
)

st.title("🏥 Medical Device Monitoring – Failure Risk Prediction (Streamlit)")
st.markdown(
    """
This app wraps your **XGBoost pipeline** and lets you simulate a single medical device
and predict its **failure risk** in real time.

Make sure your model file `xgboost_pipeline.pkl` was trained on the same kind of data
(SynDataset / SynDataset_upscaled) with matching column names.
"""
)

with st.sidebar:
    st.header("ℹ️ How to use")
    st.markdown(
        """
1. Select a **Device Name** and **Hospital Location**  
2. Adjust the numeric parameters (runtime, temperature, etc.)  
3. Click **Predict Failure Risk**  
4. See the predicted **risk level** and **confidence**

This uses the same feature names as your streaming + training scripts.
        """
    )

# --------------------------
# 4. Input controls
# --------------------------
st.subheader("1️⃣ Device & Environment Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    device_name = st.selectbox("Device Name", DEVICE_NAMES)
    device_type = DEVICE_MAPPING[device_name]  # derived from mapping
    st.text_input("Device Type (auto)", device_type, disabled=True)

with col2:
    location = st.selectbox("Hospital & Region (Location)", LOCATION_OPTIONS)
    climate_control = st.radio("Climate Control", CLIMATE_OPTIONS, horizontal=True)

with col3:
    humidity = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
    approx_age = st.slider("Approx Device Age (years)", 0.0, 40.0, 5.0, 0.5)

st.subheader("2️⃣ Operational Metrics & Sensor Readings")

c1, c2, c3 = st.columns(3)

with c1:
    runtime_hours = st.number_input(
        "Runtime Hours", min_value=0.0, max_value=20000.0, value=1000.0, step=10.0
    )
    temperature_c = st.slider(
        "Temperature (°C)", min_value=10.0, max_value=50.0, value=25.0, step=0.5
    )
    pressure_kpa = st.slider(
        "Pressure (kPa)", min_value=80.0, max_value=130.0, value=100.0, step=1.0
    )

with c2:
    vibration = st.slider(
        "Vibration (mm/s)", min_value=0.0, max_value=5.0, value=0.5, step=0.01
    )
    current_draw = st.slider(
        "Current Draw (A)", min_value=0.0, max_value=5.0, value=1.0, step=0.01
    )
    signal_noise = st.slider(
        "Signal Noise Level", min_value=0.0, max_value=10.0, value=2.0, step=0.1
    )

with c3:
    operational_cycles = st.number_input(
        "Operational Cycles", min_value=0, max_value=50000, value=5000, step=100
    )
    user_interactions = st.slider(
        "User Interactions Per Day", min_value=0.0, max_value=50.0, value=10.0, step=0.5
    )
    num_repairs = st.number_input(
        "Number of Repairs", min_value=0, max_value=50, value=1, step=1
    )
    error_logs_count = st.number_input(
        "Error Logs Count", min_value=0, max_value=100, value=3, step=1
    )

# --------------------------
# 5. Build feature vector
# --------------------------
input_record = {
    "DeviceType": device_type,
    "DeviceName": device_name,
    "RuntimeHours": float(runtime_hours),
    "TemperatureC": float(temperature_c),
    "PressureKPa": float(pressure_kpa),
    "VibrationMM_S": float(vibration),
    "CurrentDrawA": float(current_draw),
    "SignalNoiseLevel": float(signal_noise),
    "ClimateControl": climate_control,
    "HumidityPercent": float(humidity),
    "Location": location,
    "OperationalCycles": int(operational_cycles),
    "UserInteractionsPerDay": float(user_interactions),
    "ApproxDeviceAgeYears": float(approx_age),
    "NumRepairs": int(num_repairs),
    "ErrorLogsCount": int(error_logs_count),
}

# Make sure columns are in the right order
input_df = pd.DataFrame([input_record], columns=FEATURE_COLUMNS)

st.subheader("3️⃣ Prediction")

st.code(input_df.to_dict(orient="records"), language="python")

if st.button("🔮 Predict Failure Risk"):
    try:
        y_pred = pipeline.predict(input_df)[0]

        # Try to get probability if available
        proba_str = ""
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(input_df)[0]
            max_proba = float(np.max(proba))
            proba_str = f" (confidence ≈ {max_proba:.2%})"

        st.success(f"Predicted Failure Risk: **{y_pred}**{proba_str}")

        st.progress(min(1.0, (error_logs_count + num_repairs + max(0, temperature_c - 25)) / 100))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
