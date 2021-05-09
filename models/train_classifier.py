import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
# import libraries
import sys
import sqlite3
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import pickle

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
#from xgboost import XGBClassifier

import warnings
warnings.filterwarnings(action='ignore')

class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    A class to extend Sklearn's transformers to add the length of the text to the pipeline
    '''
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.Series(X).apply(lambda x: len(x)).values.reshape(-1, 1)

def load_data(data_file):
    '''
    INPUT
        data_file - database filename
    OUTPUT
        X - input variables
        y - target variables
        category_names - message types
    '''
    # load data from database
    engine = create_engine(f'sqlite:///{data_file}')
    # load to database
    df = pd.read_sql("SELECT * FROM Message", engine)
    
    # define features and label arrays
    category_names = list(df.columns[4:])
    X = df['message'].values
    y = df[category_names].values
    
    return X, y, category_names

# tokenization function to process your text data
def tokenize(text):
    '''
    INPUT
        text - one text message
    OUTPUT
        clean_tokens - clean normalised, tokenised and lemmatised text
    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    tokens = word_tokenize(text.strip())
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    """
    Instantiate pipeline
    """
    pipeline = Pipeline([
                        ('features', FeatureUnion([
                            ('txt_pipeline', Pipeline([
                                ('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer())
                            ])),
                            ('txt_len', TextLengthExtractor())
                        ])),
                        #('clf', MultiOutputClassifier(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')))
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        
                    ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
        model - trained model
        X_test - inbut test variabels
        Y_test - test labels
        category_names - message types
    '''
    # predict on test data
    Y_pred = model.predict(X_test)
    
    results_dict = dict()

    for index, col in enumerate(category_names):

        # get model results as dictionary
        report = classification_report(Y_test[:, index], Y_pred[:, index], zero_division=0, output_dict=True)
        # get results from classification report
        accuracy = report['accuracy'] 
        f1_score = report['weighted avg']['f1-score'] 
        precision = report['weighted avg']['precision'] 
        recall = report['weighted avg']['recall']
        # set result in dictionary
        results_dict[col] = {
                'Accuracy': accuracy,
                'F1-score': f1_score,
                'Precision': precision,
                'Recall': recall
        }

    results = pd.DataFrame.from_dict(results_dict, orient='index')
    print('Aggregated results: ')
    print(results.mean().to_string())
    print(results)
    
def save_model(model, model_filepath):
    """
    INPUT
        model - trained model
        model_filepath - model filename
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
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
