import os
from pathlib import Path

DATA_DIR = Path(Path(os.getcwd()).parent.parent / "data/input")
MLFLOW_HOST = "localhost"
MLFLOW_PORT = "5000"