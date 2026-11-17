
import numpy as np
import pandas as pd

def clean_categories(data):
    data = pd.DataFrame(data).copy()

    for col in data.columns:
        data[col] = data[col].map(
            lambda value: (
                str(value) if pd.notna(value) else np.nan
            )
        )

    return data
