import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from wrangle_data import download_data


def make_all_column_histograms():
    '''Loops over columns in dataset and generates histogram'''

    # download the data if not already done
    if not os.path.isdir('bank-additional'):
        download_data()

    df = pd.read_csv('bank-additional/bank-additional-full.csv', sep=';')

    # create a figures directory if not already in existence
    if not os.path.isdir('figures'):
        os.mkdir('figures')

    # iterate through each column and generate a figure
    for column in df.columns:
        if column == 'y':
            continue #skip the target variable

        make_column_histogram(df, column, 'figures/' + column + '_histogram.png')


def make_column_histogram(df, column, output_path):
    '''Creates a histogram showing the distribution of the dataset over a
    particular column, broken out by the target variable'''

    #pivot the data by the column and the target variable
    p = df.reset_index().pivot_table(values='index', index=column, columns='y',
                            aggfunc=lambda x: len(np.unique(x)), fill_value=0)

    #create histogram, adapted from
    #https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_stacked.html
    fig, ax = plt.subplots(figsize=[9.0, 6.0])

    if column in ('emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'nr.employed'):
        #some numerical columns show better as categorical
        x_ticks = list(map(str, p.index))
    elif column in ('duration', 'pdays', 'euribor3m'):
        #some columns just don't show nicely this way or aren't relevant
        return
    else:
        x_ticks = p.index

    ax.bar(x_ticks, p['yes'], label='Yes', color='green')
    ax.bar(x_ticks, p['no'], bottom=p['yes'], label='No', color='red')

    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_ylabel('Frequency')
    ax.set_xlabel(column)
    ax.set_title('Distribution of "{}" attribute broken out by target variable'.format(column))
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)



if __name__ == "__main__":

    make_all_column_histograms()
