
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from src.preprocessing import clean_categories


# --------------------------------------------------
# 1. APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Mumbai Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "best_property_model.joblib"
DEFAULTS_PATH = BASE_DIR / "models" / "input_defaults.joblib"


# --------------------------------------------------
# 2. LOAD MODEL AND DEFAULTS
# --------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_defaults():
    return joblib.load(DEFAULTS_PATH)


try:
    model = load_model()
    defaults = load_defaults()

except Exception as e:
    st.error(f"Could not load model or defaults: {e}")
    st.stop()


expected_columns = list(defaults.keys())

if hasattr(model, "feature_names_in_"):
    if list(model.feature_names_in_) != expected_columns:
        st.error("Model feature columns do not match saved defaults.")
        st.stop()


# --------------------------------------------------
# 3. APPLICATION HEADER
# --------------------------------------------------

st.title("🏠 Mumbai Property Price Predictor")

st.write(
    "Estimate Mumbai property prices using a "
    "machine-learning model."
)

st.caption(
    "Predictions are estimates based on historical "
    "property data, not guaranteed market valuations."
)


# --------------------------------------------------
# 4. CSV PREDICTION SECTION
# --------------------------------------------------

st.divider()
st.header("Predict Prices Using CSV")

uploaded_file = st.file_uploader(
    "Upload property CSV",
    type=["csv"],
    key="csv_upload"
)

if uploaded_file is not None:

    try:
        csv_input = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.write(f"Number of properties: {len(csv_input)}")
        st.dataframe(csv_input.head())

        missing_columns = [
            col for col in expected_columns
            if col not in csv_input.columns
        ]

        if missing_columns:

            st.error("Your CSV is missing required columns.")
            st.write(missing_columns)

        else:

            csv_features = csv_input[expected_columns].copy()

            if st.button("Predict CSV Prices"):

                predictions = model.predict(csv_features)

                results = csv_input.copy()
                results["Predicted Price"] = predictions

                st.subheader("CSV Prediction Results")
                st.dataframe(results)

                st.metric(
                    "Average Predicted Price",
                    f"₹{predictions.mean():,.0f}"
                )

                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Predictions",
                    data=csv_data,
                    file_name="property_predictions.csv",
                    mime="text/csv"
                )

    except Exception as e:
        st.error(f"CSV processing error: {e}")


# --------------------------------------------------
# 5. MANUAL PROPERTY INPUT SECTION
# --------------------------------------------------

st.divider()
st.header("Predict Price Using Manual Inputs")

with st.form("property_form"):

    col1, col2 = st.columns(2)

    with col1:

        covered_area = st.number_input(
            "Covered Area (sq ft)",
            min_value=100.0,
            max_value=10000.0,
            value=635.0,
            step=25.0
        )

        carpet_area = st.number_input(
            "Carpet Area (sq ft)",
            min_value=50.0,
            max_value=10000.0,
            value=375.0,
            step=25.0
        )

        floor_no = st.number_input(
            "Floor Number",
            min_value=0,
            max_value=100,
            value=5
        )

    with col2:

        developer = st.text_input(
            "Developer",
            value="Unknown"
        )

        project_name = st.text_input(
            "Project Name",
            value="Unknown"
        )

    submitted = st.form_submit_button(
        "Predict Property Price"
    )


# --------------------------------------------------
# 6. VALIDATE AND GENERATE MANUAL PREDICTION
# --------------------------------------------------

if submitted:

    if carpet_area > covered_area:

        st.error(
            "Carpet Area cannot exceed Covered Area."
        )

    else:

        property_input = {
            "Covered Area": covered_area,
            "Carpet Area": carpet_area,
            "Floor No": floor_no,
            "Developer": developer,
            "Project Name": project_name
        }

        # Start with saved training defaults
        input_row = pd.Series(defaults).copy()

        # Replace defaults with user-entered values
        for column, value in property_input.items():

            if column in input_row.index:
                input_row[column] = value

        # Create DataFrame in the correct feature order
        manual_input_df = pd.DataFrame(
            [[input_row[col] for col in expected_columns]],
            columns=expected_columns
        )

        try:

            predicted_price = float(
                model.predict(manual_input_df)[0]
            )

            if (
                not np.isfinite(predicted_price)
                or predicted_price <= 0
            ):

                st.error(
                    "The model produced an invalid price estimate."
                )

            else:

                st.success("Prediction generated successfully!")

                st.subheader("Estimated Property Price")

                st.metric(
                    "Predicted Property Price",
                    f"₹{predicted_price:,.0f}"
                )

                st.subheader("Property Details")

                st.write(
                    "Covered Area:",
                    covered_area,
                    "sq ft"
                )

                st.write(
                    "Carpet Area:",
                    carpet_area,
                    "sq ft"
                )

                st.write("Floor:", floor_no)
                st.write("Developer:", developer)
                st.write("Project:", project_name)

                st.caption(
                    "This is a machine-learning estimate. "
                    "Actual market prices may differ."
                )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )


# --------------------------------------------------
# 7. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Mumbai Property Price Predictor | "
    "Project 1 - Classical Machine Learning"
)
