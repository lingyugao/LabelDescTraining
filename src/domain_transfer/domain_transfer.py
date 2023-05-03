"""
This file is to load from a directory and test on another dataset.
"""
import os
import time
import torch
import random
import logging
import numpy as np

from importlib import reload
from utils import output_results
from config import get_args, check_args, get_model_path
from models import BaseModel, ClassifierModel
from data_process import ReadCorpus, ReadTemplates


def set_log(args):
    # set logging settings
    reload(logging)
    log_file = os.path.join(args.model_path, 'log_dt')
    if args.debug:
        handlers = [logging.FileHandler(log_file, mode='w+'), logging.StreamHandler()]
        logging.basicConfig(format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M', level=logging.INFO, handlers=handlers)
    else:
        logging.basicConfig(filename=log_file, filemode='a+',
                            level=logging.INFO, format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    logging.info(args)


def main(args):
    # read data
    corpus = ReadCorpus(args)
    dataset1 = args.dataset
    if dataset1 == 'agnews':
        dataset2 = 'yahoo'
    elif dataset1 == 'sst2':
        dataset2 = 'yelp2'
    elif dataset1 == 'sst5':
        dataset2 = 'yelp'
    elif dataset1 == 'yahoo':
        dataset2 = 'agnews'
    elif dataset1 == 'yelp2':
        dataset2 = 'sst2'
    elif dataset1 == 'yelp':
        dataset2 = 'sst5'
    else:
        raise ValueError('Not correct dataset.')
    args.data_path = args.data_path.replace(dataset1, dataset2)
    new_corpus = ReadCorpus(args)
    args.data_path = args.data_path.replace(dataset2, dataset1)

    # read templates
    templ = ReadTemplates(args)

    if args.single_line:
        if args.line_num >=0 and args.line_num < templ.temp.shape[0]:
            # line_num_list = list(range(args.line_num, templ.temp.shape[0], 1))
            if args.model_type == 'large':
                nmodel = 3
            else:
                nmodel = 3
            line_num_list = list(range(args.line_num, min(args.line_num + nmodel, templ.temp.shape[0]), 1))
            # line_num_list = [args.line_num]
        else:
            raise ValueError('line number exceeds template size!')
    else:
        if args.no_temp:
            line_num_list = [0]     # note that the template file couldn't be empty, or will cause error.
        else:
            line_num_list = list(range(templ.temp.shape[0]))

    # start running
    for line_num in line_num_list:
        # set model path
        args.pattern_type, args.pattern_pos, args.pattern_id = templ.temp.at[line_num, 'pattern type'], \
                                                               templ.temp.at[line_num, 'before/after'], \
                                                               templ.temp.at[line_num, 'pattern id']
        get_model_path(args)
        set_log(args)

        # load model
        if args.classifier:
            model = ClassifierModel(args)
        else:
            model = BaseModel(args)

        # count the total number of parameters
        pytorch_total_params = sum(p.numel() for p in model.model.parameters())
        pytorch_trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        logging.info('total para: {:d}, trainable para: {:d}'.format(pytorch_total_params, pytorch_trainable_params))

        # loading model and test
        print('log path: {:s}'.format(args.model_path))
        model.load(model.best_model_path)
        test_acc, model.avg_f1 = model.eval_model(new_corpus.test, templ.temp.loc[line_num],
                                                  cat_wise=True, df_name='domain_transfer_test_pred')
        logging.info('test acc: {:.2f}%'.format(test_acc * 100))
        output_results(args, model.best_step_size, model.best_dev_acc, model.final_dev_acc,
                       test_acc, model.final_test_acc, model.avg_f1, 'domain_transfer')


if __name__ == '__main__':
    init_time = time.time()
    args = get_args()

    args = check_args(args)

    # specify random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # this is only for 1 gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run main
    main(args)
    logging.info('total time: {:5.2f}s '.format(time.time() - init_time))
    print('finished, log path: {:s}'.format(args.model_path))




