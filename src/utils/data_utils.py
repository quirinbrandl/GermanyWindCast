import pandas as pd
from pathlib import Path

def load_dataset(resolution, split):
        x_data = pd.read_csv(
            Path(f"data/processed/{resolution}res/{split}.csv"),
            parse_dates=True,
            index_col="datetime",
        )

        return x_data