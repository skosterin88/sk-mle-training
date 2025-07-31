import os
import sys

import pickle
from pathlib import Path    

import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy as sp

from typing import List
from typing import Tuple


def apply_vectorizer(
        vectorizer: TfidfVectorizer,
        data: pl.DataFrame
) -> pl.DataFrame:
    
    corpus = data['corpus'].list.join(" ").to_numpy()
    features = vectorizer.transform(corpus)
    
    return features

def run_apply_vectorizer(
        input_frame_path: Path,
        vectorizer_path: Path,
        result_frame_path: Path,
        target_frame_path: Path
) -> None:
    
    data = pl.read_parquet(input_frame_path)
    
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    result = apply_vectorizer(vectorizer, data)

    sp.sparse.save_npz(result_frame_path, result)
    np.save(target_frame_path, data['Polarity'].to_numpy())

if __name__ == "__main__":
    
    input_frame_path = os.getcwd() + sys.argv[1]
    vectorizer_path = os.getcwd() + sys.argv[2]
    result_frame_path = os.getcwd() + sys.argv[3]
    target_frame_path = os.getcwd() + sys.argv[4]

    run_apply_vectorizer(input_frame_path, vectorizer_path, result_frame_path, target_frame_path)