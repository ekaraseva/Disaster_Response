import sys

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

# import libraries
import pandas as pd
import numpy as np
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + str(database_filepath))
    df = pd.read_sql("SELECT * FROM Data", engine)
    
    X = df['message']
    Y=df.iloc[:, 4:]
    category_names= Y.columns
    
    return X, Y, category_names


def tokenize(text):
    # remove urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    #remove punctuation
    text = re.sub(r'[^a-zA-Z09]', ' ', text)
        
    # tokenize  
    tokens = word_tokenize(text)
    
    # remove stopwords
    tokens=[w for w in tokens if w not in stopwords.words("english")]
    
    # lemmatize 
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        # on verbs
        clean_tok = lemmatizer.lemmatize(tok, pos='v').lower().strip()
        # on nouns
        clean_tok = lemmatizer.lemmatize(clean_tok).lower().strip()
        clean_tokens.append(clean_tok)
   
    # Reduce words to their stems
    clean_tokens = [PorterStemmer().stem(w) for w in clean_tokens]

    return clean_tokens

def prec_recall_report(y_true, y_pred):
    c=0
    #empty df
    scoring_results=pd.DataFrame(columns=[0, 1, 2, 3])
    for i in y_true.columns:
        prec, recall, fscore, support = precision_recall_fscore_support(
            y_true[i], y_pred[:,c], average='weighted')
        
        scoring_results=scoring_results.append(
            pd.DataFrame([i, prec, recall, fscore]).transpose())
        c=c+1
        
    scoring_results=scoring_results.rename(
        columns={0:'id', 1:'precision', 2:'recall', 
                 3:'fscore'}).reset_index(drop=True)
    return scoring_results

def f1_score_eval(y_true, y_pred):
    scoring_results=prec_recall_report(y_true, y_pred)
    
    return scoring_results['fscore'].mean()

def build_model1():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()), 
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {
        #'clf__estimator__max_depth': [10, 20, None],
        'clf__estimator__min_samples_leaf':[1, 5],
        #'clf__estimator__criterion':['gini', 'entropy']
        }
    
    scorer = make_scorer(f1_score_eval)
    cv = GridSearchCV(pipeline,  param_grid=parameters, scoring = scorer)
    
    return cv

def build_model():
    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(
            n_estimators=100, random_state=0)))
        ])
       
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred=model.predict(X_test)
    result=prec_recall_report(Y_test, Y_pred)
    print (result)
    print (result['fscore'].mean())


def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                            test_size=0.3, 
                                                            random_state=42)
        
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