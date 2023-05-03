import os
import re
import pandas as pd

data_path = '/data/origin/yahoo_answers_csv'
output_path = '/data/yahoo'
output_path2 = '/data/yahoo10'

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)
if not os.path.exists(output_path2):
    os.makedirs(output_path2, exist_ok=True)


def process_df(df):
    # we concat all these inputs
    df = df.fillna('')
    df['text'] = df['title'].str.strip() + ' ' + df['content'].str.strip() + ' ' + df['best_answer'].str.strip()
    # we only keep some of the classes
    df = df[df['label'].isin([10, 1, 6, 7, 2, 5])]
    df['label'] = df['label'].replace([2, 5], 4)
    df['label'] = df['label'].replace({10: 1, 6: 2, 7: 3})
    print(df.shape[0], df['label'].value_counts())
    return df

###############################################################################
# process train data
df_train = pd.read_csv(os.path.join(data_path, 'train.csv'),
                       names=['label', 'title', 'content', 'best_answer'], sep=',', encoding='utf-8')
df_train = process_df(df_train)


# split dev set
df_dev = df_train.groupby('label', group_keys=False).apply(
    lambda x: x.sample(frac=0.0035714, random_state=1))
df_train = df_train.drop(df_dev.index)
df_train = df_train.sample(frac=1, random_state=1).reset_index(drop=True)
df_dev = df_dev.sample(frac=1, random_state=1).reset_index(drop=True)

df_train[['label', 'text']].to_csv(os.path.join(output_path, 'train.tsv'),
                                   sep='\t', index=False, header=False, encoding='utf-8')
df_dev[['label', 'text']].to_csv(os.path.join(output_path, 'dev.tsv'),
                                 sep='\t', index=False, header=False, encoding='utf-8')

# # process test data
df_test = pd.read_csv(os.path.join(data_path, 'test.csv'),
                       names=['label', 'title', 'content', 'best_answer'], sep=',', encoding='utf-8')
df_test = process_df(df_test)
df_test[['label', 'text']].to_csv(os.path.join(output_path, 'test.tsv'),
                                  sep='\t', index=False, header=False, encoding='utf-8')

# ###############################################################################
def process_df_all(df):
    # we concat all these inputs
    df = df.fillna('')
    df['text'] = df['title'].str.strip() + ' ' + \
                 df['content'].str.strip() + ' ' + df['best_answer'].str.strip()
    print(df.shape[0], df['label'].value_counts())
    return df


# # process train data
# df_train = pd.read_csv(os.path.join(data_path, 'train.csv'),
#                        names=['label', 'title', 'content', 'best_answer'], sep=',', encoding='utf-8')
# df_train = process_df_all(df_train)
# df_train[['label', 'text']].to_csv(os.path.join(output_path2, 'train.tsv'),
#                                    sep='\t', index=False, header=False, encoding='utf-8')

# process test data
df_test = pd.read_csv(os.path.join(data_path, 'test.csv'),
                       names=['label', 'title', 'content', 'best_answer'], sep=',', encoding='utf-8')
df_test = process_df_all(df_test)
df_test[['label', 'text']].to_csv(os.path.join(output_path2, 'test.tsv'),
                                  sep='\t', index=False, header=False, encoding='utf-8')