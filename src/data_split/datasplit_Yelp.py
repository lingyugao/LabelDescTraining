import os
import re
import pandas as pd

data_path = '/data/origin/yelp_review_full_csv'
output_path = '/data/yelp'

# process train data
df_train = pd.read_csv(os.path.join(data_path, 'train.csv'),
                       names=['label', 'text'], sep=',', encoding='utf-8')
df_train['text'] = df_train['text'].str.strip()

# split dev set
df_dev = df_train.groupby('label', group_keys=False).apply(lambda x: x.sample(n=500, random_state=1))
df_train = df_train.drop(df_dev.index)
df_train = df_train.sample(frac=1, random_state=1).reset_index(drop=True)
df_train.to_csv(os.path.join(output_path, 'train.tsv'), sep='\t', index=False, header=False, encoding='utf-8')
df_dev = df_dev.sample(frac=1, random_state=1).reset_index(drop=True)
df_dev.to_csv(os.path.join(output_path, 'dev.tsv'), sep='\t', index=False, header=False, encoding='utf-8')

# process test data
df_test = pd.read_csv(os.path.join(data_path, 'test.csv'),
                       names=['label', 'text'], sep=',', encoding='utf-8')
df_test['text'] = df_test['text'].str.strip()
df_test.to_csv(os.path.join(output_path, 'test.tsv'), sep='\t', index=False, header=False, encoding='utf-8')
