"""
This file is to read log and present
precision, recall, F1 scores
"""

import os
import glob
import pandas as pd
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings; warnings.filterwarnings(action='once')


def read_df(fname, log_name='log'):
    df = pd.read_csv(os.path.join(fname, log_name),
                     engine='python', names=['text'], sep='\n\t')
    fname = os.path.basename(fname).split('_')
    df['pattern'] = fname[2]
    df['pattern_id'] = fname[3]
    df['text'] = df['text'].str.split(' ', 2).apply(lambda x: x[2])
    mark_row = df.loc[df['text'].str.startswith('test acc')].index[-1]
    df = df[mark_row - 2 * nlabel: mark_row - nlabel]
    df['text'] = df['text'].str.split(' ')
    df['label'] = df['text'].apply(lambda x: x[-9])
    df['precision'] = df['text'].apply(lambda x: x[-5].strip(',%')).astype('float')
    df['recall'] = df['text'].apply(lambda x: x[-3].strip(',%')).astype('float')
    df['F1'] = df['text'].apply(lambda x: x[-1].strip(',%')).astype('float')
    df = df[['label', 'pattern', 'pattern_id', 'precision', 'recall', 'F1']].reset_index(drop=True)
    return df

def get_new_df(df, glist):
    df1 = df.groupby(glist).agg(["mean", "std", "size"]).reset_index()
    print(df1)
    for cat in set(df.label.values):
        df2 = df.loc[df['label'] == cat]
        print(df2.shape)
        for rlabel in ['precision', 'recall', 'F1']:
            df3 = df2[rlabel].dropna().agg(["mean", "std", "size"]).reset_index()
            print('{:s}: {:.1f}\\std{{{:.1f}}}'.format(cat, df3[rlabel][0],
                                                       df3[rlabel][1]))
        print('*' * 5)

if __name__ == '__main__':
    dataset = 'yahoo'
    model_path = '/output/' + dataset
    if dataset in ['agnews', '20newsgroup', 'yahoo', 'finetune']:
        nlabel = 4
    elif dataset in ['yelp', 'sst5']:
        nlabel = 5
    elif dataset in ['sst2', 'yelp2', '20ngsim']:
        nlabel = 2
    elif dataset in ['yahoo10']:
        nlabel = 10
    # model_path = '/tmp/' + args.dataset
    # f_path_list = glob.glob(os.path.join(model_path, 'notrain_0_*base*'))
    # f_path_list = glob.glob(os.path.join(model_path, 'manual_24_*base_orig*'))
    f_path_list = glob.glob(os.path.join(model_path, 'train_40_*base_orig*'))
    for f_path in f_path_list:
        df_new = read_df(f_path)
        try:
            df_all = pd.concat([df_all, df_new], ignore_index=True)
        except NameError:
            df_all = df_new.copy()

    glist = ['label']
    get_new_df(df_all, glist)
    print('stop here')