
import joblib
import numpy as np
import pandas as pd

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "best_property_model.joblib"
DEFAULTS_PATH = BASE_DIR / "models" / "input_defaults.joblib"


def test_model_file_exists():

    assert MODEL_PATH.exists()


def test_defaults_file_exists():

    assert DEFAULTS_PATH.exists()


def test_model_generates_valid_prediction():

    model = joblib.load(MODEL_PATH)
    defaults = joblib.load(DEFAULTS_PATH)

    input_df = pd.DataFrame(
        [[defaults[col] for col in defaults.keys()]],
        columns=list(defaults.keys())
    )

    prediction = model.predict(input_df)

    assert len(prediction) == 1
    assert np.isfinite(prediction[0])
    assert prediction[0] > 0
