import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict, namedtuple
import torch.utils.data as Data

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *

def build_input_features(feature_columns):
    # Return OrderedDict: {feature_name:(start, start+dimension)}

    features = OrderedDict()

    start = 0
    for feat in feature_columns:
        feat_name = feat.name
        if feat_name in features:
            continue
        if isinstance(feat, SparseFeat):
            features[feat_name] = (start, start + 1)
            start += 1
        elif isinstance(feat, DenseFeat):
            features[feat_name] = (start, start + feat.dimension)
            start += feat.dimension
        else:
            raise TypeError("Invalid feature column type,got", type(feat))
    return features

if __name__ == "__main__":
    data = pd.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    train, test = train_test_split(data, test_size=0.2)

    train_model_input = {name: train[name] for name in feature_names}

    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []
    dense_feature_columns = list(
        filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)) if len(dnn_feature_columns) else []

    print("sparse_feature_columns:", sparse_feature_columns)

    choices = nn.ModuleDict({
        'conv': nn.Conv2d(10, 10, 3),
        'pool': nn.MaxPool2d(3)
    })

    print(choices)

    embedding_maxtrix = nn.Embedding(26,4, sparse=False)
    # this will get you a single embedding vector
    print('Getting a single vector:\n', embedding_maxtrix(torch.LongTensor([0])))
    # of course you can do the same for a seqeuncesequence
    print('Getting vectors for a sequence:\n', embedding_maxtrix(torch.LongTensor([1, 2, 3])))
    # this will give the the whole embedding matrix
    print('Getting weights:\n', embedding_maxtrix.weight.data)

    linear = False
    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=False)
         for feat in  sparse_feature_columns}
    )

    init_std = 0.0001
    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)

    feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)

    print("feature_index ", feature_index)

    target = ['label']
    X = train_model_input
    Y = train[target].values

    if isinstance(X, dict):
        X = [X[feature] for feature in feature_index]

    for i in range(len(X)):
        if len(X[i].shape) == 1:
            X[i] = np.expand_dims(X[i], axis=1)

    train_tensor_data = Data.TensorDataset(torch.from_numpy(np.concatenate(X, axis=-1)), torch.from_numpy(Y))

    torchX = torch.from_numpy(np.concatenate(X, axis=-1))
    print("torchX size: ", torchX.size())

    print("train_model_input: ", len(X))
    for feat in sparse_feature_columns:
        print("feat" , feat)
        print("feature_index[feat.name][0]" , feature_index[feat.name][0])
        test= torchX[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()
        #print("torchX range:", test)
        #print("embedding retrieved: ", embedding_dict[feat.embedding_name](test))

    sparse_embedding_list = [embedding_dict[feat.embedding_name]
            (torchX[:, feature_index[feat.name][0]:feature_index[feat.name][1]].long()) for feat in sparse_feature_columns]
    print("sparse_embedding_list:", len(sparse_embedding_list))

    dense_value_list = [torchX[:, feature_index[feat.name][0]:feature_index[feat.name][1]] for feat in dense_feature_columns]
    #print("dense_value_list:", dense_value_list)

    sparse_dnn_input = torch.flatten(
        torch.cat(sparse_embedding_list, dim=-1), start_dim=1)
    print("sparse_dnn_input size:", sparse_dnn_input.size())
    dense_dnn_input = torch.flatten(torch.cat(dense_value_list, dim=-1), start_dim=1).float()
    print("dense_dnn_input size:", dense_dnn_input.size())

    combinedInput = torch.cat([sparse_dnn_input, dense_dnn_input], dim=-1)
    print("combinedInput size:",combinedInput.size())

    #https://blog.csdn.net/Haiqiang1995/article/details/90300686
    #https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d

    #Conv2d(in_channel, out_chanel, kernel_size, stride, padding)
    conv2dlayer = nn.Conv2d(3, 2, kernel_size=3, stride = 2, padding=0)

    #input ： （batch_size , in_channel , in_height , in_width）
    inputconv2d = torch.rand(4, 3, 7, 8)

    #output: （batch_size, channel, out_height, out_width）
    outConv2d = conv2dlayer.forward(inputconv2d)

    print("outConv2d size " , outConv2d.size())

    #https://pytorch.org/docs/stable/nn.html?highlight=conv1d#torch.nn.Conv1d
    #conv1d: (in_channel, out_channel, kernel_size, stride, padding )
    conv1dlayer = nn.Conv1d(3, 5, kernel_size=3, stride=2, padding=0)

    #input: (batch_size, in_chanel, in_length)
    inputconv1d = torch.rand(24, 3, 15)

    #output: (batch_size, out_chanel, out_length)
    outConv1d = conv1dlayer.forward(inputconv1d)

    print("outConv1d size " , outConv1d.size())

    # pool of square window: https://pytorch.org/docs/stable/nn.html?highlight=avgpool1d#torch.nn.AvgPool1d
    #AvgPool2d: kernel_size, stride
    pool2dAvgModel = nn.AvgPool2d(3, stride=2)

    #Input:  (batch_size, in_channel, in_height, in_width)
    inputPool2davg = torch.randn(20, 16, 50, 32)
    #output: (batch_size, out_channel,out_height, out_width)
    outputPool2dAvg = pool2dAvgModel(inputPool2davg)
    print("outputPool2dAvg size " , outputPool2dAvg.size())

    #AvgPool2d: kernel_size, stride
    pool1dAvgModel = nn.AvgPool1d(4, stride=2)
    # Input:  (batch_size, in_channel, in_length)
    inputPool1dAvg = torch.randn(20, 16, 50)
    # output: (batch_size, out_channel,out_length)
    outputPool1dAvg = pool1dAvgModel(inputPool1dAvg)
    print("outputPool1dAvg size ", outputPool1dAvg.size())







