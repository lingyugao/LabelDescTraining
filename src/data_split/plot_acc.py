"""
This file is to read log and plot trends of dev acc for hyperparameter selection.
"""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

model_path = './output/'
f_path_list = glob.glob(os.path.join(model_path, 'manual_24*large_orig*'))

def read_df(fname):
    df = pd.read_csv(os.path.join(fname, 'log'), engine='python', names=['text'], sep='\n\t')
    df['text'] = df['text'].str.split(' ', 2).apply(lambda x: x[2])
    df = df.loc[df['text'].str.startswith('dev acc: ', na=False)]
    df['text'] = df['text'].str.split(' ').apply(lambda x: x[2])
    df['text'] = df['text'].astype('float')
    df = df.reset_index(drop=True)
    return df

save_dict = dict()
for fname in f_path_list:
    bname = os.path.basename(fname).split('_')
    lr = float(bname[-2])
    if lr in save_dict:
        save_dict[lr]['_'.join([bname[2], bname[3]])] = read_df(fname)['text'].copy()
    else:
        save_dict[lr] = read_df(fname).rename(columns={"text": '_'.join([bname[2], bname[3]])})

sep_plot = False
for lr in save_dict:
    if sep_plot:
        df.plot.line()
        plt.title('_'.join([bname[2], bname[3], str(lr), str(df.max()[0])]))
        plt.show()
        print('stop here')
    else:
        df = save_dict[lr]
        df['text'] = df.sum(axis=1)
        df['text'].plot.line()
        plt.title('_'.join([str(lr), str(df['text'].max())]))
        plt.show()
print('stop here')
