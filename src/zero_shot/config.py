import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='zero-shot')
    parser.add_argument('-nlabel', type=int, default=4, help='number of labels')

    # random seed settings
    parser.add_argument('-seed', type=int, default=1111, choices=[1111, 111, 11, 1, 42],
                        help='random seed')

    # device settings
    parser.add_argument('-device', type=str, default='gpu', help='choose cpu/gpu')

    # data settings
    parser.add_argument('-dataset', type=str, default='agnews',
                        choices=['agnews', 'yelp', 'yelp2', 'yahoo', 'yahoo10',
                                 '20newsgroup', 'sst5', 'sst2', 'finetune', '20ngsim'],
                        help='agnews, yelp, yahoo, 20newsgroup, sst5')

    # model settings
    parser.add_argument('-model_type', type=str, default='base', help='base/large')
    parser.add_argument('-train', action='store_true', help='train model')
    parser.add_argument('-manual', action='store_true', help='use manual data')
    parser.add_argument('-train_size', type=int, default=10, choices=[10, 24, 50, 100, 1000],
                        help='training data size')
    parser.add_argument('-train_type', type=str, default='all', help='training settings for manual data')

    # classifier settings
    parser.add_argument('-classifier', action='store_true', help='use classifier instead of MLM')
    parser.add_argument('-MLMclassifier', action='store_true',
                        help='use MLMclassifier, based on MLM, could not use together with classifier')
    parser.add_argument('-newclassifier', action='store_true',
                        help='MLMclassifier but only keep the first layer of MLM head, '
                             'could not use together with MLMclassifier/classifier')
    parser.add_argument('-classifier_mask', action='store_true', help='use masked position in templates')

    # label settings
    parser.add_argument('-label_type', type=str, default='orig', help='orig/shuffle/random')

    # input and output path
    parser.add_argument('-data_path', type=str, default='/data/',
                        help='location of the data sets')
    parser.add_argument('-output_data_path', type=str, default='/output/',
                        help='location of the new data sets to output')

    # template path
    parser.add_argument('-no_temp', action='store_true',
                        help='do not using any templates, this is for classifier')
    parser.add_argument('-temp_path', type=str, default='/src/zero_shot',
                        help="change here won't effect results, please put templates "
                             "in the corresponding data path, see line 85 in this file.")
    parser.add_argument('-temp_name', type=str, default='prompts.tsv',
                        help='file name of the prompts template')
    parser.add_argument('-single_line', action='store_true',
                        help='run single line or a slice from templates')
    parser.add_argument('-line_num', type=int, default=0,
                        help='line number from templates to run, works when using single_line')
    # parser.add_argument('-line_stop', type=int, default=0,
    #                     help='line number from templates to run, works when using single_line, included this line.')

    # head structure (only effective when using "-classifier -two_layer" together)
    parser.add_argument('-two_layer', action='store_true', help='use two-layer classification head')
    parser.add_argument('-add_dropout', action='store_true', help='add dropout during training')
    parser.add_argument('-use_gelu', action='store_true', help='use gelu instead of tanh')
    parser.add_argument('-layer_norm', action='store_true', help='do layer norm')

    # training settings
    parser.add_argument('-batch_size', type=int, default=1, metavar='N',
                        help='batch size for evaluation')
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('-w_decay', type=float, default=0,
                        help='weight decay for Adam')
    parser.add_argument('-training_steps', type=int, default=1000,
                        help='upper epoch limit')
    parser.add_argument('-eval_batch_size', type=int, default=32, metavar='N',
                        help='batch size for evaluation')
    parser.add_argument('-no_dev', action='store_true',
                        help='do not using dev sets')
    parser.add_argument('-freeze_half', action='store_true',
                        help='freeze half of encoder')
    parser.add_argument('-freeze_all', action='store_true',
                        help='freeze all encoder except head')

    # debug
    parser.add_argument('-debug', action='store_true',
                        help='debug mode to print outputs')

    # # save model
    # parser.add_argument('-save_model', action='store_true',
    #                     help='save best model')
    return parser.parse_args()


def check_args(args):
    # specify dataset and path
    args.data_path = os.path.join(args.data_path, args.dataset)
    args.output_data_path = os.path.join(args.output_data_path, args.dataset)
    args.temp_path = args.data_path

    # check consistency between nlabels and datasets
    wlabel = False
    if args.dataset in ['agnews', '20newsgroup', 'yahoo', 'finetune']:
        if args.nlabel != 4:
            wlabel = True
    elif args.dataset in ['yelp', 'sst5']:
        if args.nlabel != 5:
            wlabel = True
    elif args.dataset in ['sst2', 'yelp2', '20ngsim']:
        if args.nlabel != 2:
            wlabel = True
    elif args.dataset in ['yahoo10']:
        if args.nlabel != 10:
            wlabel = True
    else:
        raise ValueError('wrong dataset')
    if wlabel:
        raise ValueError('please check your label number!')
    return args


def get_model_path(args):
    assert os.path.exists(args.data_path)
    if not os.path.exists(args.output_data_path):
        os.makedirs(args.output_data_path, exist_ok=True)

    if args.classifier:
        model_path = 'classifierMask_' if args.classifier_mask else 'classifier_'
        if args.no_temp:
            if args.classifier_mask:
                raise ValueError('You could not use MASK when there is no templates.')
            args.pattern_type = 'null'
            args.pattern_id = 'null'
        if args.two_layer:
            model_path += 'twolayer_'
            if args.add_dropout:
                model_path += 'dropout_'
            if args.use_gelu:
                model_path += 'gelu_'
            if args.layer_norm:
                model_path += 'layernorm_'
    elif args.MLMclassifier:
        model_path = 'MLMclassifier_'
        if args.no_temp:
            args.pattern_type = 'null'
            args.pattern_id = 'null'
    elif args.newclassifier:
        model_path = 'newclassifier_'
        if args.no_temp:
            args.pattern_type = 'null'
            args.pattern_id = 'null'
    else:
        if args.no_temp:
            raise ValueError('You need templates for MLM!')
        else:
            model_path = ''

    if args.train:
        if args.manual:
            model_path += 'manual_' + '_'.join([str(args.train_size), args.pattern_type, str(args.pattern_id),
                                                args.model_type, args.label_type, str(args.seed),
                                                str(args.training_steps), str(args.no_dev),
                                                str(args.freeze_half), str(args.freeze_all),
                                                str(args.batch_size), str(args.lr), str(args.w_decay)])
        else:
            model_path += 'train_' + '_'.join([str(args.train_size), args.pattern_type, str(args.pattern_id),
                                               args.model_type, args.label_type, str(args.seed),
                                               str(args.training_steps), str(args.no_dev),
                                               str(args.freeze_half), str(args.freeze_all),
                                               str(args.batch_size), str(args.lr), str(args.w_decay)])
    else:
        model_path += 'notrain_' + '_'.join(['0', args.pattern_type, str(args.pattern_id), args.model_type,
                                            args.label_type, str(args.seed)])
    args.model_path = os.path.join(args.output_data_path, model_path)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path, exist_ok=True)

