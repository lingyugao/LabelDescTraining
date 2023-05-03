import os
import glob
import pandas as pd
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score
import warnings; warnings.filterwarnings(action='once')


def read_df(fname, ffname):
    df = pd.read_csv(os.path.join(fname, ffname))
    fname = os.path.basename(fname).split('_')
    new_col = fname[2] + '_' + fname[3]
    return new_col, df['pred']

def get_new_df(df):
    col_values = list(df.columns)
    for i, x in enumerate(col_values[:-1]):
        for y in col_values[i + 1:]:
            kappa = cohen_kappa_score(df[x], df[y])
            print('{:s}, {:s}, {:.2f}'.format(x, y, kappa))

if __name__ == '__main__':
    for model_size in ['base', 'large']:
        model_path = '/output/20ngsim'
        # model_path = '/tmp/' + args.dataset
        f_path_list = []
        for patterns in ['qa_1_', 'qa_2_', 'qa_3_', 'qa_4_',
                         'prompt_1_', 'prompt_2_', 'prompt_3_', 'prompt_4_', 'prompt_5_',
                         'prompt_6_', 'prompt_7_', 'prompt_8_', 'prompt_9_', 'prompt_10_']:
            f_path_list += glob.glob(os.path.join(model_path,
                                                  'notrain_0_' + patterns + '*' + model_size + '*'))

        for f_path in f_path_list:
            new_col, df_new = read_df(f_path, 'test_pred.csv')
            try:
                df_all[new_col] = df_new
            except NameError:
                df_all = df_new.to_frame(name=new_col)

        get_new_df(df_all)
