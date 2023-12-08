# The Benefits of Label-Description Training for Zero-Shot Text Classification

This repository contains the data and code for "[The Benefits of Label-Description Training for Zero-Shot Text Classification](https://arxiv.org/abs/2305.02239)." In this paper, we propose a simple way to curate small finetuning datasets intended to describe the labels for a task, which further improves zero-shot accuracies with minimal effort. 


## Data and Code

This directory includes `nlp.yml`, which could be used to create a conda environment called `nlp` by running:
```
conda env create -f nlp.yml
```

### Manual Data and Prompts

Our prompts and manual data are stored in the data directory. The required directory (e.g., data/agnews) must include  'train.tsv', 'dev.tsv', 'test.tsv' and 'manual_24.tsv' to run the code. Both train.tsv and dev.tsv could be empty files if using Label-Description training. The 'manual_24.tsv' is our manually written LabelDesc data. The 'prompts.tsv' file, containing templates for each dataset, should be placed in the same directory as the corresponding data.

For template writing, the 1st column indicates the pattern type, the 2nd column represents the pattern ID within this type, the 3rd column shows the position of the text in relation to the pattern text, and the 4th column identifies the position of mask token and whether we count from left to right or vise versa depends on the position of main text. Note that RoBERTa uses a byte-level BPE as a tokenizer.

### Code

Our code is located in the *src* directory. 
The *src/data_split* folder contains files for splitting data into train, dev, and test files. 
The *src/zero_shot* folder contains files related to Label-Description training.
The *src/domain_transfer* folder contains files for domain transfer of supervised few-shot learning, as shown in Section 4.2.3.


## Scripts

We use AGNews as an example.

To obtain zero-shot results for the MLM model:
```
python ./src/zero_shot/main.py -debug -dataset agnews -nlabel 4 -model_type base
```
To obtain results for Label-Description training with the MLM model:
```
python ./src/zero_shot/main.py -debug -dataset agnews -nlabel 4 -model_type base -train -manual -train_size 24 -lr 5e-7 -training_steps 2160 -label_type orig -freeze_half -no_dev
python ./src/zero_shot/main.py -debug -dataset agnews -nlabel 4 -model_type base -train -manual -train_size 24 -lr 5e-5 -training_steps 2160 -label_type shuffle -freeze_half -no_dev
python ./src/zero_shot/main.py -debug -dataset agnews -nlabel 4 -model_type base -train -manual -train_size 24 -lr 5e-5 -training_steps 2160 -label_type random -freeze_half -no_dev
```
To obtain results for Label-Description training with the classifier:
```
python ./src/zero_shot/main.py -debug -dataset agnews -nlabel 4 -model_type base -train -manual -train_size 24 -lr 1e-5 -training_steps 1920 -classifier -no_temp -no_dev -freeze_half -two_layer
```
To load the model from AGNews and test it on YahooAG (replace `main.py` with `domain_transfer.py`):
```
python ./src/zero_shot/domain_transfer.py -debug -dataset agnews -nlabel 4 -model_type base -train -manual -train_size 24 -lr 5e-7 -training_steps 2160 -label_type orig -freeze_half -no_dev
```

Example for getting domain transfer results (using dev set and epochs):
```
python ./src/domain_transfer/main.py -debug -dataset agnews -nlabel 4 -seed 11 -model_type base -train -train_size 400 -lr 1e-5 -epochs 15 -label_type orig -freeze_half -batch_size 2
```

## Citation
```
@inproceedings{gao-etal-2023-benefits,
    title = "The Benefits of Label-Description Training for Zero-Shot Text Classification",
    author = "Gao, Lingyu  and
      Ghosh, Debanjan  and
      Gimpel, Kevin",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.853",
    pages = "13823--13844"
}
```
