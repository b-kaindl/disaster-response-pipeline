import sys
from typing import Callable
import pandas as pd

from sqlalchemy import create_engine, Engine
from pandas import DataFrame, Series

def load_data(messages_filepath: str, categories_filepath: str) -> DataFrame:

    msgs: DataFrame = pd.read_csv(messages_filepath)
    cats: DataFrame = pd.read_csv(categories_filepath)

    df: DataFrame = pd.merge(msgs, cats, on='id')

    return df


def clean_data(df: DataFrame) -> DataFrame:
    # create a dataframe of the 36 individual category columns
    cats: DataFrame = df.categories.str.split(';', expand=True)

    # use first row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    row: Series = cats.iloc[0, :]
    get_colnames: Callable[[str], str] = lambda x: x[:-2]
    category_colnames: Series = row.map(get_colnames)

    cats.columns = category_colnames

    for column in cats:
    # set each value to be the last character of the string
        cats[column] = cats[column].astype(str).str[-1]

    # convert column from string to numeric
        cats[column] = cats[column].astype(int)

    df = df.drop(columns='categories').join(cats).drop_duplicates()

    # remove useless values
    df = df.drop(index=df.loc[df.related == 2].index)

    return df

def save_data(df: DataFrame, database_filename: str) -> None:
    engine: Engine = create_engine('database_filename')
    df.to_sql('dp.messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
