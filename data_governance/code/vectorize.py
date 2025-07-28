import os
import sys

import pickle
from pathlib import Path

import dvc

import dvc.api
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Tuple


def train_vectorize(
        data: pl.DataFrame
) -> Tuple[TfidfVectorizer, pl.DataFrame, pl.DataFrame]:
    
    random_state = 42
    params = {
        'max_features': 10000,
        'analyzer': 'word'
        }
    
    # params = dvc.api.params_show()
    print(params)
    tfidf_vectorizer = TfidfVectorizer(**params)
    # tfidf_vectorizer = TfidfVectorizer(**params["vectorizer_tfidf"])

    train, test = train_test_split(
        data,
        test_size=0.3,
        shuffle=True,
        random_state=random_state)

    tfidf_vectorizer.fit(train['corpus'].list.join(" ").to_numpy())

    return tfidf_vectorizer, train, test


def vectorize_train(input_frame_path: Path,
                    vectorizer_path: Path,
                    train_features_path: Path,
                    test_features_path: Path
                    ) -> None:
    
    data = pl.read_parquet(input_frame_path)
    # print(len(data))
    # print(data.columns)
    # print(data[0]['corpus'].item())
    vectorizer, train, test = train_vectorize(data)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    train.write_parquet(train_features_path)
    test.write_parquet(test_features_path)

if __name__ == "__main__":
    
    input_frame_path = os.getcwd() + sys.argv[1]
    vectorizer_path = os.getcwd() + sys.argv[2]
    train_features_path = os.getcwd() + sys.argv[3]
    test_features_path = os.getcwd() + sys.argv[4]

    vectorize_train(input_frame_path, vectorizer_path, train_features_path, test_features_path)