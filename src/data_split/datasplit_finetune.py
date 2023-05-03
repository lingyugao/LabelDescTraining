"""
This file is to create a dataset to get hyperparameters
We use manual data as training data, 20newsgroup as dev set to tune parameters.
"""
import os
import pandas as pd
from cleantext import clean
from sklearn.datasets import fetch_20newsgroups


output_path = '/data/finetune'
if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

def clean_df(df):
    for id in range(df.shape[0]):
        df.at[id, 'text'] = clean(df.at[id, 'text'], lower=False,
                                  no_line_breaks=True)
    return df.loc[df['text'] != '']

def get_df(subset):
    categories = ['rec.autos', 'sci.med', 'talk.politics.guns', 'talk.religion.misc']
    newsgroups_df = fetch_20newsgroups(subset=subset,
                                       remove=('headers', 'footers', 'quotes'),
                                       categories=categories)
    assert(newsgroups_df.target_names == categories)
    df = pd.DataFrame(newsgroups_df.data, columns=['text'])
    df['label'] = newsgroups_df.target
    df['label'] = df['label'] + 1
    df = clean_df(df)
    return df

df = get_df('all')

# output dev
df[['label', 'text']].to_csv(os.path.join(output_path, 'dev.tsv'),
                             sep='\t', index=False, header=False, encoding='utf-8')

