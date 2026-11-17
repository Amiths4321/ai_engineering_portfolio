
import numpy as np
import pandas as pd

from src.preprocessing import clean_categories


def test_converts_numbers_to_strings():

    data = pd.DataFrame({
        "Developer": ["ABC Builders", 123, "XYZ Developers"]
    })

    result = clean_categories(data)

    assert result["Developer"].tolist() == [
        "ABC Builders",
        "123",
        "XYZ Developers"
    ]


def test_preserves_missing_values():

    data = pd.DataFrame({
        "Developer": ["ABC Builders", np.nan, "XYZ Developers"]
    })

    result = clean_categories(data)

    assert pd.isna(result["Developer"].iloc[1])


def test_handles_empty_dataframe():

    data = pd.DataFrame()

    result = clean_categories(data)

    assert result.empty
