import sys
import sqlite3
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(file_path_messages, file_path_categories):
    
    # read datasets
    messages = pd.read_csv(file_path_messages,encoding='utf-8')
    categories = pd.read_csv(file_path_categories,encoding='utf-8')
    
    # Merge datasets
    df = pd.merge(messages,categories,how='inner',on='id')
    
    # Create df with 36 categories
    categories = df['categories'].str.split(';',expand=True)
    
    # Extract list of titles for df of 36 categories
    row = categories.iloc[0,:]
    
    # Extract list of names of categories and Rename the headers of df categories
    category_colnames = row.str.split('-',expand=True)[0]
    categories.columns = category_colnames
    
    # Convert categories to 0-1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    categories.replace(2, 1, inplace=True)
        
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1,sort=True)
    
    return df

def clean_data(df):
    # drop duplicates
    df = df.drop_duplicates()
    return df


def load_to_database(df, database_filename):
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('Disaster_data', engine,if_exists = 'replace', index=False)   


def main():
    if len(sys.argv) == 4:

        file_path_messages, file_path_categories, database_filename = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(file_path_messages, file_path_categories))
        df = load_data(file_path_messages, file_path_categories)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filename))
        load_to_database(df, database_filename)
        
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