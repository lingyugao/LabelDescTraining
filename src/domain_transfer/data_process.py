import os
import re
import pandas as pd


class ReadCorpus(object):
    def __init__(self, args):
        self.params = args
        if not self.params.manual:
            self.train = self.read_df('train.tsv', is_train=True)
        else:
            self.manual_24 = self.read_df('manual_24.tsv', is_train=True, is_manual=True)
        # self.manual_40 = self.read_df('manual_40.tsv', is_train=True, is_manual=True)
        # self.manual_80 = self.read_df('manual_80.tsv', is_train=True, is_manual=True)
        if self.params.train and (not self.params.no_dev):
            self.dev = self.read_df('dev.tsv')
        else:
            self.dev = pd.DataFrame([])
        self.test = self.read_df('test.tsv')

    def proc_df(self, df):
        df['text'] = df['text'].apply(self.clean_str)
        # df['description'] = df['description'].apply(self.clean_str)
        df['label'] = df['label'].astype(int)
        return df

    def read_df(self, fname, is_train=False, is_manual=False):
        df = pd.read_csv(os.path.join(self.params.data_path, fname),
                         names=['label', 'text', 'description'], sep='\t', encoding='utf-8')
        df = self.proc_df(df)
        if is_train and self.params.train:
            if is_manual:
                df = df.sample(frac=1, random_state=self.params.seed).reset_index(drop=True)
            else:
                df = df.sample(frac=1, random_state=self.params.seed).reset_index(drop=True)
                if self.params.no_sample:
                    if self.params.dataset in ['yahoo', 'agnews']:
                        self.params.train_size = 8544
                    elif self.params.dataset in ['sst2', 'yelp2']:
                        self.params.train_size = 6920
                    elif self.params.dataset in ['sst5', 'yelp5']:
                        self.params.train_size = 8544
                quotient = self.params.train_size // self.params.nlabel
                remainder = self.params.train_size % self.params.nlabel
                if remainder != 0:
                    df_1 = df.groupby('label', group_keys=False).apply(
                        lambda x: x.sample(quotient, random_state=self.params.seed))
                    df_2 = df.drop(df_1.index).groupby('label').apply(lambda x: x.sample(1, random_state=self.params.seed))
                    df_2 = df_2.sample(remainder, random_state=self.params.seed)
                    df = pd.concat([df_1, df_2]).sample(frac=1, random_state=self.params.seed).reset_index(drop=True)
                else:
                    df = df.groupby('label').apply(lambda x: x.sample(quotient, random_state=self.params.seed))
                    df = df.sample(frac=1, random_state=self.params.seed).reset_index(drop=True)
        return df

    @staticmethod
    def clean_str(input_str):
        input_str = re.sub(r"\s{2,}", " ", input_str)  # replace two or more whitespaces with one whitespace
        input_str = (
            input_str.replace("\\\\", "\\")
        )
        return input_str.strip()


class ReadTemplates(object):
    def __init__(self, args):
        self.params = args
        self.temp = pd.read_csv(os.path.join(args.temp_path, args.temp_name), sep='\t', encoding='latin-1')
        self.clean_temp()

    def clean_temp(self):
        """
        file include:pattern type, pattern id, before/after, string, mask position
        strip string columns
        """
        df_obj = self.temp.select_dtypes(['object'])
        self.temp[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
        self.temp[['pattern id', 'mask position']] = self.temp[['pattern id', 'mask position']].apply(pd.to_numeric)
        if (self.temp[self.temp['before/after'] == 'before']['mask position'] >= 0).any():
            raise ValueError('Mask position should be negative when using "before"')
        if (self.temp[self.temp['before/after'] == 'after']['mask position'] < 0).any():
            raise ValueError('Mask position should be non-negative when using "after"')
