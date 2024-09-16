
import re
from pathlib import Path

import click
import nltk
import polars as pl
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .cli import cli





def preprocess_text(input_text: str) -> str:

    text = input_text.lower()
    
    # delete hyperlinks
    text = re.sub(r'https?://\S+|www\.\S+|\[.*?\]|[^a-zA-Z\s]+|\w*\d\w*', "", text)
    # delete special symbols
    text = re.sub("[0-9 \-_]+", " ", text)
    # leave letters only
    text = re.sub("^[a-z A-Z]+")
    # delete stopwords
    text = " ".join([word for word in text.split() if word not in stopwords.words("english")])

    return text.strip()

def lemmatize(input_frame: pl.DataFrame) -> pl.DataFrame:

    lemmatizer = WordNetLemmatizer()

    return input_frame.with_columns(pl.with_column(
        pl.col('corpus').map_elements(
            lambda input_list: [lemmatizer.lemmatize(token) for token in input_list])
        )
    )

def preprocess_dataframe(data: pl.DataFrame, col_name: str) -> pl.DataFrame:

    return lemmatize(
        data.with_columns(
            pl.col(col_name)
            .map_elements(preprocess_text)
            .str.split(" ")
            .alias("corpus")
        )
    )

# Run preprocessing from the console for DVC to be able to include this in dvc repro pipeline
@cli.command()
@click.argument("input_frame_path", type=Path)
@click.argument("output_frame_path", type=Path)
@click.argument("column", type=str, default="Review")
def cli_preprocessing(input_frame_path: Path, 
                      output_frame_path: Path, 
                      column: str) -> None:

    nltk.download('stopwords')

    # read raw data
    data = pl.read_csv(
        input_frame_path,
        has_header=False,
        new_columns=['Polarity', 'Title', 'Review'], 
        n_rows=50000
    )
    # preprocess specified column
    processed_data = preprocess_dataframe(data, column)
    # save the data to parquet file
    processed_data.write_parquet(output_frame_path)
