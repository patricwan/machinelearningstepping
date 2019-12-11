
# coding: utf-8

# In[ ]:

import torch
from torchtext import data

SEED = 1234

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

TEXT = data.Field(tokenize='spacy')
LABEL = data.LabelField(dtype=torch.float)


# In[2]:

#https://towardsdatascience.com/use-torchtext-to-load-nlp-datasets-part-i-5da6f1c89d84
from torchtext import datasets

import logging

import pandas as pd
import numpy as np

VAL_RATIO = 0.2
def prepare_csv(trainCsvPath, testCsvPath, seed=999):
    df_train = pd.read_csv(trainCsvPath)
    df_train["comment_text"] =         df_train.comment_text.str.replace("\n", " ")
    idx = np.arange(df_train.shape[0])
    np.random.seed(seed)
    np.random.shuffle(idx)
    val_size = int(len(idx) * VAL_RATIO)
    df_train.iloc[idx[val_size:], :].to_csv(
        "cache/dataset_train.csv", index=False)
    df_train.iloc[idx[:val_size], :].to_csv(
        "cache/dataset_val.csv", index=False)
    
    df_test = pd.read_csv(testCsvPath)
    df_test["comment_text"] =         df_test.comment_text.str.replace("\n", " ")
    df_test.to_csv("cache/dataset_test.csv", index=False)
    
import re
import spacy
NLP = spacy.load('en')
MAX_CHARS = 20000
def tokenizer(comment):
    comment = re.sub(
        r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", 
        str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    if (len(comment) > MAX_CHARS):
        comment = comment[:MAX_CHARS]
    return [
        x.text for x in NLP.tokenizer(comment) if x.text != " "]


import torch
from torchtext import data
LOGGER = logging.getLogger("toxic_dataset")
def get_dataset(trainCsvPath, testCsvPath, fix_length=100, lower=False, vectors=None):
    if vectors is not None:
        # pretrain vectors only supports all lower cases
        lower = True
    LOGGER.debug("Preparing CSV files...")
    prepare_csv(trainCsvPath, testCsvPath)
    comment = data.Field(
        sequential=True,
        fix_length=fix_length,
        tokenize=tokenizer,
        pad_first=True,
        #tensor_type=torch.cuda.LongTensor,
        lower=lower
    )
    LOGGER.debug("Reading train csv file...")
    train, val = data.TabularDataset.splits(
        path='cache/', format='csv', skip_header=True,
        train='dataset_train.csv', validation='dataset_val.csv',
        fields=[
            ('id', None),
            ('comment_text', comment),
            ('toxic', data.Field(
                use_vocab=False, sequential=False)), #tensor_type=torch.cuda.ByteTensor
            ('severe_toxic', data.Field(
                use_vocab=False, sequential=False)), #tensor_type=torch.cuda.ByteTensor
            ('obscene', data.Field(
                use_vocab=False, sequential=False)), #tensor_type=torch.cuda.ByteTensor
            ('threat', data.Field(
                use_vocab=False, sequential=False)), #tensor_type=torch.cuda.ByteTensor
            ('insult', data.Field(
                use_vocab=False, sequential=False)), #tensor_type=torch.cuda.ByteTensor
            ('identity_hate', data.Field(
                use_vocab=False, sequential=False
                )),   #tensor_type=torch.cuda.ByteTensor
        ])
    LOGGER.debug("Reading test csv file...")
    test = data.TabularDataset(
        path='cache/dataset_test.csv', format='csv', 
        skip_header=True,
        fields=[
            ('id', None),
            ('comment_text', comment)
        ])
    LOGGER.debug("Building vocabulary...")
    comment.build_vocab(
        train, val, test,
        max_size=20000,
        min_freq=50,
        vectors=vectors
    )
    LOGGER.debug("Done preparing the datasets")
    return train, val, test

def get_iterator(dataset, batch_size, train=True, 
    shuffle=True, repeat=False):
    dataset_iter = data.Iterator(
        dataset, batch_size=batch_size, device=0,
        train=train, shuffle=shuffle, repeat=repeat,
        sort=False
    )
    return dataset_iter


dataSet = get_dataset("../../../data/nlp/jigsawtoxic/train.csv", "../../../data/nlp/jigsawtoxic/test.csv")


# In[3]:

#print(dataSet.shape)


# In[ ]:

import torchtext
from torchtext import data

# tokenizer function using spacy
nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
def tokenizer(s): 
    return [w.text.lower() for w in nlp(tweet_clean(s))]

def tweet_clean(text):
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text) # remove non alphanumeric character
    text = re.sub(r'https?:/\/\S+', ' ', text) # remove links
    return text.strip()

# define the columns that we want to process and how to process
txt_field = data.Field(sequential=True, 
                       tokenize=tokenizer, 
                       include_lengths=True, 
                       use_vocab=True)
label_field = data.Field(sequential=False, 
                         use_vocab=False, 
                         pad_token=None, 
                         unk_token=None)

train_val_fields = [
    ('ItemID', None), # we dont need this, so no processing
    ('Sentiment', label_field), # process it as label
    ('SentimentSource', None), # we dont need this, so no processing
    ('SentimentText', txt_field) # process it as text
]

trainData = data.TabularDataset.splits(path='../../../data/nlp/', 
                                            format='csv', 
                                            train='Sentiment.csv',
                                            fields=train_val_fields, 
                                            skip_header=True)

print(len(trainData))
#print(len(validationData))

from torchtext import vocab
# specify the path to the localy saved vectors
vec = vocab.Vectors('glove.6B.50d.txt', '../../../data/nlp/')
# build the vocabulary using train and validation dataset and assign the vectors
txt_field.build_vocab(trainData, max_size=100000, vectors=vec)
# build vocab for labels
label_field.build_vocab(trainData)

print(txt_field.vocab.vectors.shape)
# torch.Size([100002, 100])

txt_field.vocab.vectors[txt_field.vocab.stoi['the']]




