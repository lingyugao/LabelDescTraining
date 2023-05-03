"""
This file is to read test pred and compute:
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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import warnings; warnings.filterwarnings(action='once')


def read_df(fname, ffname):
    df = pd.read_csv(os.path.join(fname, ffname))
    fname = os.path.basename(fname).split('_')
    lname = list(set(df['gold'].values))
    # print(df['gold'].value_counts())
    prec, recall, f1, label_num = precision_recall_fscore_support(df['gold'], df['pred'], average=None,
                                                                  labels=lname, zero_division=0)
    accuracy = accuracy_score(df['gold'], df['pred'])
    new_df = pd.DataFrame(columns=['label', 'precision', 'recall', 'F1'])

    for i, label in enumerate(lname):
        new_df = new_df.append({'label': label, 'precision': prec[i] * 100,
                                'recall': recall[i] * 100, 'F1': f1[i] * 100}, ignore_index=True)
    new_df['pattern'] = fname[2]
    new_df['pattern_id'] = fname[3]
    return new_df, accuracy

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
    dataset = 'yahoo10'
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
    # f_path_list = glob.glob(os.path.join(model_path, 'notrain_0_*large*'))
    f_path_list = glob.glob(os.path.join(model_path, 'manual_24_*base_orig*'))
    # f_path_list = glob.glob(os.path.join(model_path, 'train_2000_*large_orig*'))
    acc = []
    for f_path in f_path_list:
        df_new, accuracy = read_df(f_path, 'test_pred.csv')
        # df_new, accuracy = read_df(f_path, 'domain_transfer_test_pred.csv')
        try:
            df_all = pd.concat([df_all, df_new], ignore_index=True)
        except NameError:
            df_all = df_new.copy()
        acc.append(accuracy)

    glist = ['label']
    get_new_df(df_all, glist)
    print('accuracy: {:.1f}\\std{{{:.1f}}}'.format(np.average(np.array(acc)) * 100,
                                                   np.std(np.array(acc)) * 100))