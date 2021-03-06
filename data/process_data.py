#import libraries
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    ''' Function to load data 
    
    INPUT: 
    messages_filepath - messages.csv file file containing original messages
    categories_filepath - categories.csv file containing allocation of messages
    per disaster category 
    
    OUTPUT: 
    df - combained dataframe containing messages + categories
    '''
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, how='outer', on='id')
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    #convert category values to  0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.extract('(\d+)')
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)    
    return df

def clean_data(df): 
    ''' Cleaning function
    
    INPUT: df - dataframe containing messages and categories where duplicates, 
    empty columns and very short messages (#NAME)
    
    OUTPUT: df - cleaned dataframe
    '''
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    df=df.reset_index(drop=True)
    
    # drop related = 2 values
    df=df.drop(df[df['related']==2]['message'].index).reset_index(drop=True)
    
    # drop child alone column 
    df=df.drop(['child_alone'], axis=1)
    
    # remove extra short messages
    df=df.drop(df[df['message'].str.len()<15].index).reset_index(drop=True)
    
    return df    


def save_data(df, database_filename):
    '''Save data to database'''
    
    engine = create_engine('sqlite:///' + str(database_filename))
    df.to_sql('Data', engine, if_exists='replace', index=False)
    pass  


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