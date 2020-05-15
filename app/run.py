import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
import re
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
import nltk
app = Flask(__name__)


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-z0-9]"," ",text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    clean_tokens = []
    lemmatizer = WordNetLemmatizer()
    
    for w in words:
        clean_tok = lemmatizer.lemmatize(w , pos='v').strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def tokenize_2(text):
    """ Tokenize input text. This function is called in StartingVerbExtractor. """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        """ Return true if the first word is an appropriate verb or RT for retweet """
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize_2(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        """ Fit """
        return self

    def transform(self, X):
        """ Transform """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql("SELECT * FROM DisasterTable", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    category_names = df.iloc[:, 4:].columns
    category_counts = (df.iloc[:, 4:]).sum(
    ).sort_values(ascending=False).values

    # Calculate message count by genre and related status
    related = df[df['related'] == 1].groupby('genre').count()[
        'message']
    not_related = df[df['related'] == 0].groupby('genre').count()[
        'message']
    graphs = [
        # GRAPH 3 - related status
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=related,
                    name='Related'
                ),

                Bar(
                    x=genre_names,
                    y=not_related,
                    name='Not Related'
                )
            ],

            'layout': {
                'title': 'Distribution of Messages by Genre and Related Status',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode': 'stack'
            }
        },
        
        # GRAPH 2 - category graph
        {
            'data': [
                Bar(
                    x=category_names[:5],
                    y=category_counts[:5]
                )
            ],

            'layout': {
                'title': 'Distribution of Top 5 Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': 35
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
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