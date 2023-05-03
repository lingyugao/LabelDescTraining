import os
import torch
import copy
import math
import collections
import logging
import transformers
import torch.nn as nn
import pandas as pd

from utils import *
from sklearn.metrics import confusion_matrix
from transformers import RobertaTokenizer, RobertaForMaskedLM, RobertaConfig
from model_utils import RobertaForSequenceClassification


class BaseModel(object):
    def __init__(self, args):
        self.params = args
        self.device = get_device(args.device)
        self.config = 'roberta-' + self.params.model_type
        self.t = RobertaTokenizer.from_pretrained(self.config)
        if not self.params.no_temp:
            if self.params.pattern_pos == 'before':
                self.t.truncation_side = 'left'
        self.mask_id = self.t.encode('<mask>', add_special_tokens=False)[0]
        self.get_model()

        # default to use MLM loss
        self.crit = nn.CrossEntropyLoss(ignore_index=1)
        self.opt = torch.optim.Adam(
            params=filter(
                lambda p: p.requires_grad, self.model.parameters()
            ),
            lr=self.params.lr, weight_decay=self.params.w_decay,)
        self.get_word_list()
        self.best_model_path = os.path.join(args.model_path, 'best_model.pt')
        self.final_model_path = os.path.join(args.model_path, 'final_model.pt')
        self.best_step_size = 0
        self.best_dev_acc, self.final_dev_acc, self.final_test_acc = 0., 0., 0.
        self.freeze_para()

    def compute_prf1(self, y_list, pred_list, llabel, lname, df_name='test_pred'):
        prec, recall, f1, label_num = precision_recall_fscore_support(y_list, pred_list, average=None,
                                                                      labels=llabel, zero_division=0)
        for i, label in enumerate(lname):
            logging.info('{} num {:.0f}, precision: {:.2f}%, recall: {:.2f}%, F1: {:.2f}%'.format(
                label, label_num[i], prec[i] * 100, recall[i] * 100, f1[i] * 100))
        avg_f1 = np.average(f1)
        results_df = pd.DataFrame({'pred': pred_list, 'gold': y_list}, columns=['pred', 'gold'])
        refer_dict = {x: y for (x, y) in zip(llabel, lname)}
        results_df.replace(refer_dict, inplace=True)
        results_df.to_csv(os.path.join(self.params.model_path, df_name + '.csv'),
                          header=True, index=False, mode='w+')
        cm = confusion_matrix(y_list, pred_list, labels=llabel)
        logging.info(cm)
        np.savetxt(os.path.join(self.params.model_path, df_name + '_cm.txt'), cm, fmt='%d')
        return avg_f1

    def freeze_para(self):
        if self.params.freeze_all:
            for name, param in self.model.named_parameters():
                if 'lm_head' not in name:
                    param.requires_grad = False
        elif self.params.freeze_half:
            half_layer_num = self.model.config.num_hidden_layers / 2
            for name, param in self.model.named_parameters():
                if 'roberta.encoder.layer.' in name:
                    name = name.split('.')
                    if int(name[3]) < half_layer_num:
                        param.requires_grad = False
                # elif 'roberta.embeddings' in name:
                #     param.requires_grad = False

    def specify_labels(self):
        if self.params.dataset in ['agnews', 'yahoo']:
            return [' World', ' Sports', ' Business', ' Tech']
        elif self.params.dataset in ['yahoo10']:
            return [' Society', ' Science', ' Health', ' Education', ' Computer',
             ' Sports', ' Business', ' Entertainment', ' Relationship', ' Politics']
            # return ['culture', 'science', 'health', 'education', 'computer',
            #         'sports', 'business', 'music', 'family', 'politics']
        elif self.params.dataset in ['yelp', 'sst5']:
            return [' terrible', ' bad', ' okay', ' good', ' great']
        elif self.params.dataset in ['sst2', 'yelp2']:
            return [' awful', ' great']
        elif self.params.dataset in ['20ngsim']:
            return [' PC', ' Mac']
        elif self.params.dataset in ['20newsgroup', 'finetune']:
            return [' automobile', ' medicine', ' gun', ' religion']
        else:
            raise ValueError('No suitable labels.')

    def check_maskid(self, x, x_mask):
        """
        This is to check whether the template input is correct on <mask> position for the 1st sentence.
        :return:
        """
        if self.mask_pos < 0:
            mask_pos_1d = x_mask.sum(axis=1) + self.mask_pos
            if int(mask_pos_1d) == x.tolist()[0].index(self.mask_id):
                print('Mask position correct!')
            else:
                raise ValueError('Wrong mask position.')
        else:
            if self.mask_pos == x.tolist()[0].index(self.mask_id):
                print('Mask position correct!')
            else:
                raise ValueError('Wrong mask position.')

    def get_model(self):
        """
        load MLM. If use newclassifier, random initialize MLM head layer.
        """
        if self.params.newclassifier:
            self.crit = nn.CrossEntropyLoss()
            # re-initialize the decoder layer (linear layer)
            config = RobertaConfig.from_pretrained(self.config)
            config.tie_word_embeddings = False
            self.model = RobertaForMaskedLM.from_pretrained(self.config, config=config)
            self.model.lm_head.decoder = nn.Linear(config.hidden_size, self.params.nlabel)
            self.model.lm_head.bias = nn.Parameter(torch.zeros(self.params.nlabel))
            self.model.lm_head.decoder.bias = nn.Parameter(torch.zeros(self.params.nlabel))
            self.model.to(self.device)
            self.model.lm_head.decoder.weight.data.normal_(mean=0.0, std=config.initializer_range)
            self.model.lm_head.decoder.bias.data.zero_()
            self.model.lm_head.bias.data.zero_()
            self.word_list = self.specify_labels()
        else:
            self.model = RobertaForMaskedLM.from_pretrained(self.config).to(self.device)
            if self.params.label_type == 'orig':
                self.word_list = self.specify_labels()
            elif self.params.label_type == 'shuffle':
                word_list = self.specify_labels()
                self.word_list = derangement_shuffle(word_list, copy.copy(word_list))
            elif self.params.label_type == 'random':
                self.word_list = ['RANDOM' + str(i) for i in range(self.params.nlabel)]
                self.t.add_tokens(self.word_list)
                self.model.resize_token_embeddings(len(self.t))
                logging.info('Add vocab: %s'% list(self.model.roberta.embeddings.word_embeddings.weight[-4:, :].size()))
                self.model.roberta.embeddings.word_embeddings.weight[-4:, :]\
                    .data.normal_(mean=0.0, std=self.model.config.initializer_range)
            else:
                raise ValueError('label type error')
            logging.info(self.word_list)

    def encode_label(self, word):
        output = self.t.encode(word, add_special_tokens=False)
        assert(len(output) == 1)
        return output[0]

    def get_word_list(self):
        """
        Specify classes (verbalizer)
        """
        # self.word_list = ['World', 'Sports', 'Business', 'Tech']
        if self.params.newclassifier:
            self.word_enc_list = [i for i in range(self.params.nlabel)]
        else:
            self.word_enc_list = []
            for word in self.word_list:
                self.word_enc_list.append(self.encode_label(word))

    def proc_data(self, df, temp_df, print_example=False, istrain=False):
        if not (self.params.classifier or self.params.newclassifier):
            df = df.replace({"label": {i + 1: x for i, x in enumerate(self.word_list)}})
        # if self.params.manual and istrain:
        #     df['text'] = df['title']
        # else:
        #     df['text'] = df['title'] + ' ' + df['description']

        if not self.params.no_temp:
            if self.params.MLMclassifier or self.params.newclassifier:
                self.mask_pos = 0
            else:
                self.mask_pos = temp_df['mask position']
                self.mask_pos = self.mask_pos - 1 if self.mask_pos < 0 else self.mask_pos + 1  # consider sos and eos

            if temp_df['before/after'] == 'before':
                df['text'] = df['text'] + ' ' + temp_df['string']
            elif temp_df['before/after'] == 'after':
                df['text'] = temp_df['string'] + ' ' + df['text']
            else:
                raise ValueError('Please check before/after in template file!')
        else:
            self.mask_pos = 0

        if print_example:
            logging.info('Data Example: {}, label {}'.format(df.at[1, 'text'], df.at[1, 'label']))
            if not (self.params.no_temp or self.params.MLMclassifier
                    or self.params.newclassifier or self.params.classifier):
                sequences = self.t([df.at[1, 'text']], padding=True,
                                   truncation=True, add_special_tokens=True,
                                   max_length=256, return_tensors='pt')
                self.check_maskid(sequences.input_ids, sequences.attention_mask)
        return df

    def get_batch(self, df, bsz):
        length = df.shape[0]
        logging.info('data set size: {:d}, batch size: {:d}'.format(length, bsz))
        nbatch = (length + bsz - 1) // bsz

        if self.params.newclassifier:
            # padding and return batch
            output = []
            for idx in range(nbatch):
                start_idx, end_idx = idx * bsz, min((idx + 1) * bsz, length)
                sequences = self.t(df['text'].iloc[start_idx: end_idx].tolist(), padding=True,
                                   truncation=True, add_special_tokens=True, max_length=256, return_tensors='pt')
                y = (torch.LongTensor(df['label'].iloc[start_idx: end_idx].tolist()) - 1).to(self.device)
                output.append(
                    [sequences.input_ids.to(self.device), sequences.attention_mask.to(self.device), y.view(-1)])
            return output
        else:
            # padding and return batch
            output = []
            for idx in range(nbatch):
                start_idx, end_idx = idx * bsz, min((idx + 1) * bsz, length)
                sequences = self.t(df['text'].iloc[start_idx: end_idx].tolist(), padding=True,
                                   truncation=True, add_special_tokens=True, max_length=256, return_tensors='pt')
                y = self.t(df['label'].iloc[start_idx: end_idx].tolist(), padding=False,
                           truncation=False, add_special_tokens=False, return_tensors='pt').input_ids.to(self.device)
                output.append([sequences.input_ids.to(self.device), sequences.attention_mask.to(self.device), y.view(-1)])
            return output

    def train_model(self, train_data, dev_data, test_data, temp_df):
        """
        dev data is not needed here, only to match classifier input format
        """
        self.model.train()
        train_data = self.get_batch(self.proc_data(train_data, temp_df, print_example=True,
                                                   istrain=True), self.params.batch_size)
        total_step = 0.
        dev_acc, best_dev_acc, best_step_size, epoch = 0., 0., 0., 1
        best_train_loss = math.inf
        while(total_step < self.params.training_steps):
            corrects, total_size, total_loss = 0., 0., 0.
            for x, x_mask, y in train_data:
                if total_step < self.params.training_steps:
                    # if total_step % 500 == 0 and total_step >= 500:
                    #     self.final_test_acc, self.avg_f1 = self.eval_model(test_data, temp_df, cat_wise=True,
                    #                                           df_name='test_pred_' + str(total_step))
                    #     logging.info('test acc: {:.2f}%'.format(self.final_test_acc * 100))
                    #     output_results(self.params, self.best_step_size, self.best_dev_acc,
                    #                    self.final_dev_acc, 0., self.final_test_acc, self.avg_f1,
                    #                    'output_' + str(total_step))
                    if self.mask_pos < 0:
                        mask_pos_1d = x_mask.sum(axis=1) + self.mask_pos
                        output_all = self.model(input_ids=x, attention_mask=x_mask).logits
                        mask_pos = mask_pos_1d.view(-1, 1, 1).expand(x.shape[0], 1, output_all.shape[2])
                        output = output_all.gather(1, mask_pos).view(x.shape[0], -1)
                    else:
                        output = self.model(input_ids=x, attention_mask=x_mask).logits[:, self.mask_pos, :]
                    loss = self.crit(output, y)
                    total_loss += loss.data
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    corrects += float((output.argmax(dim=1) == y).sum())
                    total_step += x.size()[0]
                    total_size += x.size()[0]
                else:
                    break
            epoch += 1
            logging.info('Epoch {:d}, train acc: {:.2f} %, train loss: {:f}, data size: {:.0f}'.format(
                epoch - 1, corrects / total_size * 100, total_loss, total_size))
            if not self.params.no_dev:
                dev_acc, _ = self.eval_model(dev_data, temp_df)
                logging.info('dev acc: {:.2f} %'.format(dev_acc * 100))
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_step_size = total_step
                    self.save(self.best_model_path)
        # self.save(self.final_model_path)
        # self.load(self.best_model_path)
        logging.info('best step size: {:.0f}'.format(best_step_size))
        self.best_step_size = best_step_size
        self.best_dev_acc = best_dev_acc
        self.final_dev_acc = dev_acc
        return

    @torch.no_grad()
    def eval_model(self, test_data, temp_df, cat_wise=False, df_name='test_pred'):
        self.model.eval()
        test_data = self.get_batch(self.proc_data(test_data, temp_df), self.params.eval_batch_size)
        pred_list, y_list = torch.empty(size=(0,)).to(self.device), torch.empty(size=(0,)).to(self.device)
        for x, x_mask, y in test_data:
            if self.mask_pos < 0:
                mask_pos_1d = x_mask.sum(axis=1) + self.mask_pos
                output_all = self.model(input_ids=x, attention_mask=x_mask).logits
                mask_pos = mask_pos_1d.view(-1, 1, 1).expand(x.shape[0], 1, output_all.shape[2])
                output = output_all.gather(1, mask_pos).view(x.shape[0], -1)
            else:
                mask_pos = self.mask_pos
                output = self.model(input_ids=x, attention_mask=x_mask).logits[:, mask_pos, :]
            chose_token = torch.index_select(output, 1, torch.tensor(self.word_enc_list).to(self.device))
            pred = torch.LongTensor([self.word_enc_list[x] for x in chose_token.argmax(dim=1)]).to(self.device)
            pred_list = torch.cat([pred_list, pred.detach().clone()], dim=0)
            y_list = torch.cat([y_list, y.detach().clone()], dim=0)
        self.model.train()
        corrects = float((pred_list == y_list).sum())
        total_size = y_list.size()[0]
        avg_f1 = 0.
        if cat_wise:
            avg_f1 = self.compute_prf1(y_list.cpu().numpy(), pred_list.cpu().numpy(),
                                       self.word_enc_list, self.word_list, df_name=df_name)
            # df_final = pd.DataFrame([corrects / total_size * 100], columns=['test_acc'])
            # df_final.to_csv(os.path.join(self.params.model_path, 'test_acc.csv'))
        return corrects / total_size, avg_f1

    @torch.no_grad()
    def eval_model_yahoo(self, test_data, temp_df, cat_wise=False, df_name='test_pred'):
        """
        inference for YahooAG/AGNews with Yahoo10 model
        """
        self.model.eval()

        # remove unused labels, construct new correspondance
        self.word_enc_list = self.word_enc_list[:2] + self.word_enc_list[4:7] + self.word_enc_list[9:10]
        self.pred_list = []
        for word in ['World', 'Tech', 'Tech', 'Sports', 'Business', 'World']:
            self.pred_list.append(self.encode_label(word))

        test_data = self.get_batch(self.proc_data(test_data, temp_df), self.params.eval_batch_size)
        pred_list, y_list = torch.empty(size=(0,)).to(self.device), torch.empty(size=(0,)).to(self.device)
        for x, x_mask, y in test_data:
            if self.mask_pos < 0:
                mask_pos_1d = x_mask.sum(axis=1) + self.mask_pos
                output_all = self.model(input_ids=x, attention_mask=x_mask).logits
                mask_pos = mask_pos_1d.view(-1, 1, 1).expand(x.shape[0], 1, output_all.shape[2])
                output = output_all.gather(1, mask_pos).view(x.shape[0], -1)
            else:
                mask_pos = self.mask_pos
                output = self.model(input_ids=x, attention_mask=x_mask).logits[:, mask_pos, :]
            chose_token = torch.index_select(output, 1, torch.tensor(self.word_enc_list).to(self.device))
            pred = torch.LongTensor([self.pred_list[x] for x in chose_token.argmax(dim=1)]).to(self.device)
            pred_list = torch.cat([pred_list, pred.detach().clone()], dim=0)
            y_list = torch.cat([y_list, y.detach().clone()], dim=0)
        self.model.train()
        corrects = float((pred_list == y_list).sum())
        total_size = y_list.size()[0]
        avg_f1 = 0.

        # get new correspondance
        self.word_list = ['World', 'Tech', 'Sports', 'Business']
        self.word_new_enc = []
        for word in self.word_list:
            self.word_new_enc.append(self.encode_label(word))

        if cat_wise:
            avg_f1 = self.compute_prf1(y_list.cpu().numpy(), pred_list.cpu().numpy(),
                                       self.word_new_enc, self.word_list, df_name=df_name)
            # df_final = pd.DataFrame([corrects / total_size * 100], columns=['test_acc'])
            # df_final.to_csv(os.path.join(self.params.model_path, 'test_acc.csv'))
        return corrects / total_size, avg_f1

    def save(self, save_path):
        self.best_dict = self.model.state_dict().copy()
        checkpoint = {
            'model_state_dict': self.best_dict,
        }
        self.checkpoint = checkpoint
        # torch.save(checkpoint, save_path)

    def save_current(self, save_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict().copy(),
        }
        torch.save(checkpoint, save_path)

    def load(self, save_path):
        try:
            self.best_dict
            torch.save(self.checkpoint, save_path)
            self.model.load_state_dict(self.best_dict)
            self.model.to(self.device)
        except AttributeError:
            checkpoint = torch.load(save_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
        logging.info('model loaded')


class ClassifierModel(BaseModel):
    def __init__(self, args):
        super(ClassifierModel, self).__init__(args)
        self.crit = nn.CrossEntropyLoss()

    def get_model(self):
        config = RobertaConfig.from_pretrained(self.config)
        self.model = RobertaForSequenceClassification(self.config, config=config, params=self.params,
                                                      classifier_mask=self.params.classifier_mask,
                                                      num_labels=self.params.nlabel).to(self.device)
        self.word_list = self.specify_labels()

    def get_batch(self, df, bsz):
        length = df.shape[0]
        logging.info('data set size: {:d}, batch size: {:d}'.format(length, bsz))
        nbatch = (length + bsz - 1) // bsz

        # padding and return batch
        output = []
        for idx in range(nbatch):
            start_idx, end_idx = idx * bsz, min((idx + 1) * bsz, length)
            sequences = self.t(df['text'].iloc[start_idx: end_idx].tolist(), padding=True,
                               truncation=True, add_special_tokens=True, max_length=256, return_tensors='pt')
            y = (torch.LongTensor(df['label'].iloc[start_idx: end_idx].tolist()) - 1).to(self.device)
            output.append([sequences.input_ids.to(self.device), sequences.attention_mask.to(self.device), y.view(-1)])
        return output

    def train_model(self, train_data, dev_data, test_data, temp_df):
        self.model.train()
        train_data = self.get_batch(self.proc_data(train_data, temp_df, print_example=True,
                                                   istrain=True), self.params.batch_size)
        total_step = 0.
        dev_acc, best_dev_acc, best_step_size, epoch = 0., 0., 0., 1
        best_train_loss = math.inf
        while(total_step < self.params.training_steps):
            corrects, total_size, total_loss = 0., 0., 0.
            for x, x_mask, y in train_data:
                if total_step < self.params.training_steps:
                    # if total_step % 500 == 0 and total_step >= 500:
                    #     self.final_test_acc, self.avg_f1 = self.eval_model(test_data, temp_df, cat_wise=True,
                    #                                           df_name='test_pred_' + str(total_step))
                    #     logging.info('test acc: {:.2f}%'.format(self.final_test_acc * 100))
                    #     output_results(self.params, self.best_step_size, self.best_dev_acc,
                    #                    self.final_dev_acc, 0., self.final_test_acc, self.avg_f1,
                    #                    'output_' + str(total_step))
                    output = self.model(input_ids=x, attention_mask=x_mask,
                                        mask_pos=self.mask_pos).logits
                    loss = self.crit(output, y)
                    total_loss += loss.data
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                    corrects += float((output.argmax(dim=1) == y).sum())
                    total_step += x.size()[0]
                    total_size += x.size()[0]
                else:
                    break
            epoch += 1
            logging.info('Epoch {:d}, train acc: {:.2f} %, train loss: {:f}, data size: {:.0f}'.format(
                epoch - 1, corrects / total_size * 100, total_loss, total_size))
            if not self.params.no_dev:
                dev_acc, _ = self.eval_model(dev_data, temp_df)
                logging.info('dev acc: {:.2f} %'.format(dev_acc * 100))
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    best_step_size = total_step
                    self.save(self.best_model_path)
        # self.save(self.final_model_path)
        # self.load(self.best_model_path)
        logging.info('best step size: {:.0f}'.format(best_step_size))
        self.best_step_size = best_step_size
        self.best_dev_acc = best_dev_acc
        self.final_dev_acc = dev_acc
        return

    @torch.no_grad()
    def eval_model(self, test_data, temp_df, cat_wise=False, df_name='test_pred'):
        self.model.eval()
        test_data = self.get_batch(self.proc_data(test_data, temp_df), self.params.eval_batch_size)
        pred_list, y_list = torch.empty(size=(0,)).to(self.device), torch.empty(size=(0,)).to(self.device)
        for x, x_mask, y in test_data:
            output = self.model(input_ids=x, attention_mask=x_mask,
                                mask_pos=self.mask_pos).logits
            pred_list = torch.cat([pred_list, output.argmax(dim=1).detach().clone()], dim=0)
            y_list = torch.cat([y_list, y.detach().clone()], dim=0)
        self.model.train()
        corrects = float((pred_list == y_list).sum())
        total_size = y_list.size()[0]
        avg_f1 = 0.
        if cat_wise:
            print(collections.Counter(pred_list.tolist()))
            avg_f1 = self.compute_prf1(y_list.cpu().numpy(), pred_list.cpu().numpy(),
                                       list(range(self.params.nlabel)), self.word_list, df_name=df_name)
        return corrects/total_size, avg_f1
