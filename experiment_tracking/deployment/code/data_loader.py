import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(Path(os.getcwd()))

from constants import DATA_DIR

def read_data(filename: str) -> pd.DataFrame:
    """
    Read source data.

    Args:
        filename: str - filename to read data from.

    Returns:
        pd.DataFrame - dataframe with source data.
    """
   
    if filename.endswith('.parquet'):
        df = pd.read_parquet(DATA_DIR / filename)
    elif filename.endswith('.zip'):
        df = pd.read_csv(DATA_DIR / filename, compression='zip')
    else:
        df = pd.read_csv(DATA_DIR / filename)

    df = df.set_index('Datetime')
    df.index = pd.to_datetime(df.index)

    return df