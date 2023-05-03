import os
import pandas as pd
import numpy as np
from datasets import load_dataset


output_path = '/data/sst5'
output_path2 = '/data/sst2'

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
if not os.path.exists(output_path2):
    os.makedirs(output_path, exist_ok=True)

def process_df(df):
    df = df.rename(columns={"sentence": "text", "label": "score"})
    # we concat all these inputs
    criteria = [df['score'].between(0, 0.2, inclusive='both'),
                df['score'].between(0.2, 0.4, inclusive='right'),
                df['score'].between(0.4, 0.6, inclusive='right'),
                df['score'].between(0.6, 0.8, inclusive='right'),
                df['score'].between(0.8, 1, inclusive='right')]
    values = [1, 2, 3, 4, 5]
    df['label'] = np.select(criteria, values, 0)
    print(df.shape[0], df['label'].value_counts())
    return df

# load the sentiment treebank corpus in the parenthesis format,
sst_train = load_dataset("sst", split="train").to_pandas()
df_train = process_df(sst_train)
df_train[['label', 'text']].to_csv(os.path.join(output_path, 'train.tsv'),
                                   sep='\t', index=False, header=False, encoding='utf-8')

sst_dev = load_dataset("sst", split="validation").to_pandas()
df_dev = process_df(sst_dev)
df_dev[['label', 'text']].to_csv(os.path.join(output_path, 'dev.tsv'),
                                 sep='\t', index=False, header=False, encoding='utf-8')

sst_test = load_dataset("sst", split="test").to_pandas()
df_test = process_df(sst_test)
df_test[['label', 'text']].to_csv(os.path.join(output_path, 'test.tsv'),
                                  sep='\t', index=False, header=False, encoding='utf-8')

# load SST-2
df_train = load_dataset("gpt3mix/sst2", split="train").to_pandas()
print(df_train.shape[0], df_train['label'].value_counts())
df_train['label'] = df_train['label'].replace({0: 2})
df_train[['label', 'text']].to_csv(os.path.join(output_path2, 'train.tsv'),
                                   sep='\t', index=False, header=False, encoding='utf-8')

df_dev = load_dataset("gpt3mix/sst2", split="validation").to_pandas()
print(df_dev.shape[0], df_dev['label'].value_counts())
df_dev['label'] = df_dev['label'].replace({0: 2})
df_dev[['label', 'text']].to_csv(os.path.join(output_path2, 'dev.tsv'),
                                 sep='\t', index=False, header=False, encoding='utf-8')


df_test = load_dataset("gpt3mix/sst2", split="test").to_pandas()
print(df_test.shape[0], df_test['label'].value_counts())
df_test['label'] = df_test['label'].replace({0: 2})
df_test[['label', 'text']].to_csv(os.path.join(output_path2, 'test.tsv'),
                                  sep='\t', index=False, header=False, encoding='utf-8')

