import requests
from shutil import unpack_archive
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os


def download_data():
    '''Download source Bank Marketing Data Set'''

    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"

    with open('bank-additional.zip', 'wb') as f:
        f.write(requests.get(data_url).content)

    unpack_archive('bank-additional.zip')


def clean_data():
    '''Returns a cleaned version of the dataset ready for model training'''

    # download the data if not already done
    if not os.path.isdir('bank-additional'):
        download_data()

    df = pd.read_csv('bank-additional/bank-additional-full.csv', sep=';')

    # Categorical variables need to be broken into one-hot representations

    categorical_columns = [
        'job',
        'marital',
        'education',
        'default',
        'housing',
        'loan',
        'contact',
        'month',
        'day_of_week',
        'poutcome'
    ]

    for column in categorical_columns:
        df = make_one_hot_column(df, column)
        df.drop(columns=column, inplace=True) #drop original column

    # special case because contact is a binary column, we don't need both
    df.drop(columns='contact_telephone', inplace=True)

    # also convert the binary target variable to zeroes and ones
    df['y'] = df['y'].transform(lambda y: 0 if y == 'no' else 1)

    # The documentation accompanying the data notes that "the duration is not
    # known before a call is performed" and thus "should be discarded if the
    # intention is to have a realistic predictive model". We want to have a
    # realistic predictive model so we will drop this column

    df.drop(columns='duration', inplace=True)

    # >96% of clients have never previously been contacted (i.e. pdays = 999),
    # making the pdays column junky and redundant with previous and poutcome.
    # There are also some contradictory records where pdays = 999 and yet
    # previous > 0 and poutcome is something other than 'nonexistent'. Since
    # pdays is mostly null and apparently untrustworthy (previous and poutcome
    # are consistent with each other), let's just drop it.

    df.drop(columns='pdays', inplace=True)

    # There's something important about the overall volume of client contacts
    # during a given month. Let's create a feature for this. The cons.price.idx
    # attribute is calculated monthly and looks precise enough to be practically
    # unique per month, we'll use this as a proxy to group by unique month.

    to_add = df[['cons.price.idx']].reset_index().groupby(['cons.price.idx']).count().rename(columns={'index':'total_clients_contacted_in_month'})
    df = df.join(to_add, on='cons.price.idx')

    # This will be the dataset that we'll use for training!
    return df


def make_one_hot_column(df, column):
    '''Takes a categorical column and appends new columns representing a one-hot
    encoding of the column to the dataframe. New columns will be labeled as
    [original column]_[category]'''

    encoder = OneHotEncoder()
    encoder.fit(df[[column]])

    #create new column labels
    new_cols = [column + '_' + cat for cat in encoder.categories_[0]]

    #append new one-hot columns to dataframe
    transformed = encoder.transform(df[[column]]).toarray()
    one_hot_data = pd.DataFrame(transformed, columns=new_cols)
    df = pd.concat([df, one_hot_data], axis=1)

    return df



if __name__ == "__main__":

    # save a cleaned version of the dataset, creating a directory if not exists
    if not os.path.isdir('data'):
        os.mkdir('data')

    df = clean_data()
    df.to_csv('data/cleaned_dataset.csv')
