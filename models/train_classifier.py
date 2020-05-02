import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet') # download for lemmatization
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

from pandas import DataFrame
from typing import List

def load_data(database_filepath: str) -> DataFrame:
    engine = create_engine('sqlite:///' + database_filepath)
    df: DataFrame = pd.read_sql_table('dp.messages', engine)

    x_cols: str = 'message'
    X: DataFrame = df[x_cols].astype(str)
    Y: DataFrame = df.drop(columns = [x_cols, 'id', 'original', 'genre'])
    return X,Y,Y.columns


def tokenize(text: str) -> List[str]:
#     normalize
    text = re.sub(r'[^a-zA-z0-9]',' ', text.lower())

    words = [word for word in word_tokenize(text) if word not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()

    clean_tokens : List[str] = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model() -> GridSearchCV:
    pipeline : Pipeline = Pipeline([
    ('textpipe', CountVectorizer(tokenizer = tokenize)), # put TFIDF here if necessary
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs = -1)))])

    parameters = {'clf__estimator__criterion': ['gini', 'entropy'],
    'clf__estimator__max_depth' : [125,150,175],
    'clf__estimator__n_estimators' : [1,5],
    'clf__estimator__min_samples_leaf' : [5,10]}

    cv : GridSearchCV = GridSearchCV(pipeline, param_grid=parameters, cv = 3,
    verbose=2, scoring='f1_weighted')
    # cv = pipeline
    return cv

def evaluate_model(model, X_test, Y_test, category_names) -> None:

    Y_pred = model.predict(X_test)
    print(classification_report(Y_test.values,Y_pred,target_names=category_names))


def save_model(model, model_filepath: str) -> None:
    with open(model_filepath, 'wb') as file:
        pickle.dump(model,file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
