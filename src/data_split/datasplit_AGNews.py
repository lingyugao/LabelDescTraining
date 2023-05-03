import os
import re
import pandas as pd

data_path = './data/origin/ag_news_csv'
output_path = './data/agnews'

# process train data
df_train = pd.read_csv(os.path.join(data_path, 'train.csv'),
                       names=['label', 'title', 'description'], sep=',', encoding='utf-8')
df_train['text'] = df_train['title'].str.strip() + ' ' + df_train['description'].str.strip()
df_train = df_train.drop(columns=['title', 'description'])

# split dev set
df_dev = df_train.groupby('label', group_keys=False).apply(lambda x: x.sample(n=500, random_state=1))
df_train = df_train.drop(df_dev.index)
df_train = df_train.sample(frac=1, random_state=1).reset_index(drop=True)
df_train.to_csv(os.path.join(output_path, 'train.tsv'), sep='\t', index=False, header=False, encoding='utf-8')
df_dev = df_dev.sample(frac=1, random_state=1).reset_index(drop=True)
df_dev.to_csv(os.path.join(output_path, 'dev.tsv'), sep='\t', index=False, header=False, encoding='utf-8')

# process test data
df_test = pd.read_csv(os.path.join(data_path, 'test.csv'),
                       names=['label', 'title', 'description'], sep=',', encoding='utf-8')
df_test['text'] = df_test['title'].str.strip() + ' ' + df_test['description'].str.strip()
df_test = df_test.drop(columns=['title', 'description'])
df_test.to_csv(os.path.join(output_path, 'test.tsv'), sep='\t', index=False, header=False, encoding='utf-8')
