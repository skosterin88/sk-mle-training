from sys import argv

import pickle
from pathlib import Path

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from typing import Tuple


def train_vectorize(
        data: pl.DataFrame
) -> Tuple[TfidfVectorizer, pl.DataFrame, pl.DataFrame]:
    
    random_state = 42
    vectorizer_params = {'max_features': 10000,
                         'analyzer': 'word'
                         }
    
    tfidf_vectorizer = TfidfVectorizer(**vectorizer_params)

    train, test = train_test_split(
        data,
        test_size=0.3,
        shuffle=True,
        random_state=random_state
    )

    tfidf_vectorizer.fit(train['corpus'].list.join(" ").to_numpy())

    return tfidf_vectorizer, train, test


def vectorize_train(input_frame_path: Path,
                    vectorizer_path: Path,
                    train_features_path: Path,
                    test_features_path: Path
                    ) -> None:
    
    data = pl.read_parquet(input_frame_path)
    print(len(data))
    print(data.columns)
    print(data[0]['corpus'])
    vectorizer, train, test = train_vectorize(data)
    pickle.dump(vectorizer, vectorizer_path.open('wb'))

    train.write_parquet(train_features_path)
    test.write_parquet(test_features_path)


input_frame_path = Path(argv[1])
vectorizer_path = Path(argv[2])
train_features_path = Path(argv[3])
test_features_path = Path(argv[4])

vectorize_train(input_frame_path, 
                vectorizer_path, 
                train_features_path, 
                test_features_path)