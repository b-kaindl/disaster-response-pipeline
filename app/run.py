import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.subplots import make_subplots
from plotly.graph_objs import Bar, Scatter

from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from typing import List


app = Flask(__name__)

def tokenize(text: str) -> List[str]:
    """
    Tokenize function from train_classifier.py
    Normaliztes text data and creates responding tokens for NLP models
    """
#     normalize
    text = re.sub(r'[^a-zA-z0-9]',' ', text.lower())

    words = [word for word in word_tokenize(text) if word not in stopwords.words('english')]

    lemmatizer = WordNetLemmatizer()

    clean_tokens : List[str] = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# load data
database_filepath = './data/DisasterResponse.db'
engine = create_engine('sqlite:///' + database_filepath)
df = pd.read_sql_table('dp.messages', engine)

# HACK: split dataset for performance stats
x_cols: str = 'message'
X = df[x_cols].astype(str)
Y = df.drop(columns = [x_cols, 'id', 'original', 'genre'])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

# load model
model_cv = joblib.load('./models/classifier_10p_for_test.pkl')
model_nocv = joblib.load('./models/classifier_nocv.pkl')

# compute performance
Y_pred_cv = model_cv.predict(X_test)
Y_pred_nocv = model_nocv.predict(X_test)

report_cv = pd.DataFrame.from_dict(classification_report(Y_test.values,Y_pred_cv,
target_names=Y.columns, output_dict=True), 'index')

report_nocv = pd.DataFrame.from_dict(classification_report(Y_test.values,Y_pred_nocv,
target_names=Y.columns, output_dict=True), 'index')


labels = report_cv.index.values#.iloc[:-4,:]
# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    performance_plot = make_subplots()
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            # TODO: can improve this by grouping by model/
            # need concat with column indicating group
            'data': [
                Bar(
                    name = 'Support',
                    x=labels,
                    y=report_nocv['support'].iloc[:-4],
                    yaxis='y1'

                ),

                Scatter(
                    name = 'Grid Search Opt.',
                    x=labels,
                    y=report_nocv['f1-score'],
                    yaxis='y2'

                ),

                # Scatter(
                #     name = 'W.o. Grid Search Opt.',
                #     x=labels,
                #     y=report_nocv['f1-score'],
                #     yaxis='y2'
                # )
            ],

            'layout': {
                'title': 'Support and F1 Score by Label - W.o Grid Search ',
                'yaxis1': {
                    'title': "Messages",
                    'side' : 'left',
                    'tickformat' :','
                },
                'yaxis2': {
                    'title': "F1 Score",
                    'side' : 'right',
                    'tickformat' :'%',
                    'overlaying': 'y'
                },
                'xaxis': {
                    'title': "Label"
                },
                # 'barmode' : 'grouped'
            },


        },

        {
            # TODO: can improve this by grouping by model/
            # need concat with column indicating group
            'data': [
                Bar(
                    name = 'Support',
                    x=labels,
                    y=report_cv['support'].iloc[:-4],
                    yaxis='y1'

                ),

                Scatter(
                    name = 'F1 Score',
                    x=labels,
                    y=report_cv['f1-score'],
                    yaxis='y2'

                ),

            ],

            'layout': {
                'title': 'Support and F1 Score by Label - Grid Search Opt.',
                'yaxis1': {
                    'title': "Messages",
                    'side' : 'left',
                    'tickformat' :','
                },
                'yaxis2': {
                    'title': "F1 Score",
                    'side' : 'right',
                    'tickformat' :'%',
                    'overlaying': 'y'
                },
                'xaxis': {
                    'title': "Label"
                },
                # 'barmode' : 'grouped'
            },

            # 'secondary_y' : [False,True,True]
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotFalsely graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
