import os
import random
import torch
import socket
import logging

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


# Device configuration(cpu/gpu), get machine number
def get_device(set_device):
    if set_device == 'gpu':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    logging.info('=' * 89)
    logging.info('device type: {:s}, hostname: {:s}'.format(str(device), socket.gethostname()))
    logging.info('=' * 89)

    return device


def derangement_shuffle(cmp, lst):
    """
    make sure no element in same position
    """
    random.shuffle(lst)
    for old, new in zip(cmp, lst):
        if old == new:
            return derangement_shuffle(cmp, lst)
    return lst


# save output results to file
def output_results(args, best_step_size: int, best_dev_acc: float, final_dev_acc: float,
                   test_acc: float, final_test_acc: float, avg_f1: float, output_name: str):
    out_dic = vars(args)
    out_dic['best_dev_acc'] = best_dev_acc
    out_dic['best_step_size'] = best_step_size
    out_dic['test_acc'] = test_acc
    out_dic['final_dev_acc'] = final_dev_acc
    out_dic['final_test_acc'] = final_test_acc
    out_dic['avg_f1'] = avg_f1
    out_df = pd.DataFrame.from_dict(out_dic, orient='index').transpose()

    out_df.to_csv(os.path.join(args.model_path, output_name + '.csv'),
                  header=True, index=False, mode='w+')
