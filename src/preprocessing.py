import pandas as pd

def clean_categories(data):
    data = pd.DataFrame(data).copy()

    for col in data.columns:
        data[col] = data[col].map(
            lambda value: str(value) if pd.notna(value) else float("nan")
        )

    return data