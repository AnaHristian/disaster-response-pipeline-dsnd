import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    A class to extend Sklearn's transformers to add the length of the text to the pipeline
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Given an input text, return its length
        """
        return pd.Series(X).apply(lambda x: len(x)).values.reshape(-1, 1)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # graph 1
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    graph_one = []
    
    graph_one.append(
        Bar(
            x = genre_names,
            y = genre_counts
            )
        )
      
    layout_one = dict(title = 'Distribution of Message Genres',
                    xaxis = dict(title = 'Count'),
                    yaxis = dict(title = 'Genre'),
                )
    
    # graph 2 - top ten messages
    categories = list(df.columns[4:])
    message_name = list(df[categories].sum().sort_values(ascending=False)[:10].index)
    message_percent = list((df[categories].sum().sort_values(ascending=False)/df.shape[0]*100).iloc[:10])
    
    graph_two = []
    
    graph_two.append(
      Bar(
      x = message_percent,
      y = message_name,
      orientation = 'h'
      )
    )

    layout_two = dict(title = 'Top Ten Message Types',
                xaxis = dict(title = 'Message Type',),
                yaxis = dict(title = 'Messages %'),
                )


    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    
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
