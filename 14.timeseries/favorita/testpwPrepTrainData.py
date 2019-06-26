import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers
import gc
import os

from Utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings

df, promo_df, items, stores = load_unstack('pw')

# data after 2015
df = df[pd.date_range(date(2017,1,1), date(2017,8,15))]
promo_df = promo_df[pd.date_range(date(2017,1,1), date(2017,8,15))]

promo_df = promo_df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
df = df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
promo_df = promo_df.astype('int')

df_test = pd.read_csv("../../../data/favgrocery/test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df.index.get_level_values(1)

item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))
df = df.loc[df.index.get_level_values(1).isin(item_inter)]
promo_df = promo_df.loc[promo_df.index.get_level_values(1).isin(item_inter)]

df_index = df.index
del item_nbr_test, item_nbr_train, item_inter, df_test; gc.collect()

print("df head", df.columns)
print("promo-df ", promo_df.columns)

print("items", items.head)
print("stores", stores.head)

def train_generator(df, promo_df, items, stores, timesteps, first_pred_start,
    n_range=1, day_skip=7, is_train=True, batch_size=2000, aux_as_tensor=False, reshape_output=0, first_pred_start_2017=None):
    encoder = LabelEncoder()
    dfLevelValues = df.index.get_level_values(1)
    print("dfLevelValues " , dfLevelValues)
    
    items_reindex = items.reindex(dfLevelValues)
    print("items_reindex " , items_reindex)
    
    item_family = encoder.fit_transform(items_reindex['family'].values)
    item_class = encoder.fit_transform(items_reindex['class'].values)
    item_perish = items_reindex['perishable'].values

    stores_reindex = stores.reindex(df.index.get_level_values(0))
    store_nbr = df.reset_index().store_nbr.values - 1
    store_cluster = stores_reindex['cluster'].values - 1
    store_type = encoder.fit_transform(stores_reindex['type'].values)

    item_group_mean = df.groupby('item_nbr').mean()
    store_group_mean = df.groupby('store_nbr').mean()

    cat_features = np.stack([item_family, item_class, item_perish, store_nbr, store_cluster, store_type], axis=1)

    print("cat_features " , cat_features)
    
    while 1:
        date_part = np.random.permutation(range(n_range))
        print("random date_part " , date_part)
        #if first_pred_start_2017 is not None:
        #    range_diff = (first_pred_start - first_pred_start_2017).days / day_skip
        #    date_part = np.concat([date_part, np.random.permutation(range(range_diff, int(n_range/2) + range_diff))])

        for i in date_part:
            keep_idx = np.random.permutation(df.shape[0])[:batch_size]
            print("keep_idx  " , keep_idx)
            df_tmp = df.iloc[keep_idx,:]
            promo_df_tmp = promo_df.iloc[keep_idx,:]
            cat_features_tmp = cat_features[keep_idx]
            pred_start = first_pred_start + timedelta(days=int(day_skip*i))
            print("first_pred_start " , first_pred_start)
            print("pred_start " , pred_start)

            # Generate a batch of random subset data. All data in the same batch are in the same period.
            yield create_dataset_part(df_tmp, promo_df_tmp, cat_features_tmp, item_group_mean, store_group_mean, timesteps, pred_start, reshape_output, aux_as_tensor, True)

            gc.collect()

def create_dataset_part(df, promo_df, cat_features, item_group_mean, store_group_mean, timesteps, pred_start, reshape_output, aux_as_tensor, is_train, weight=False):

    item_mean_df = item_group_mean.reindex(df.index.get_level_values(1))
    store_mean_df = store_group_mean.reindex(df.index.get_level_values(0))
    # store_family_mean_df = store_family_group_mean.reindex(df.index)

    X, y = create_xy_span(df, pred_start, timesteps, is_train)
    is0 = (X==0).astype('uint8')
    promo = promo_df[pd.date_range(pred_start+timedelta(days=timesteps), periods=timesteps+1)].values
    weekday = np.tile([d.weekday() for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
                          (X.shape[0],1))
    dom = np.tile([d.day-1 for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
                          (X.shape[0],1))
    item_mean, _ = create_xy_span(item_mean_df, pred_start, timesteps, False)
    store_mean, _ = create_xy_span(store_mean_df, pred_start, timesteps, False)
    yearAgo, _ = create_xy_span(df, pred_start-timedelta(days=365), timesteps+16, False)
    quarterAgo, _ = create_xy_span(df, pred_start-timedelta(days=91), timesteps+16, False)

    if reshape_output>0:
        X = X.reshape(-1, timesteps, 1)
    if reshape_output>1:
        is0 = is0.reshape(-1, timesteps, 1)
        promo = promo.reshape(-1, timesteps+16, 1)
        yearAgo = yearAgo.reshape(-1, timesteps+16, 1)
        quarterAgo = quarterAgo.reshape(-1, timesteps+16, 1)
        item_mean = item_mean.reshape(-1, timesteps, 1)
        store_mean = store_mean.reshape(-1, timesteps, 1)
        # store_family_mean = store_family_mean.reshape(-1, timesteps, 1)

    w = (cat_features[:, 2] * 0.25 + 1) / (cat_features[:, 2] * 0.25 + 1).mean()

    cat_features = np.tile(cat_features[:, None, :], (1, timesteps+16, 1)) if aux_as_tensor else cat_features

    # Use when only 6th-16th days (private periods) are in the training output
    # if is_train: y = y[:, 5:]

    if weight: return ([X, is0, promo, yearAgo, quarterAgo, weekday, dom, cat_features, item_mean, store_mean], y, w)
    else: return ([X, is0, promo, yearAgo, quarterAgo, weekday, dom, cat_features, item_mean, store_mean], y)

def create_xy_span(df, pred_start, timesteps, is_train=True, shift_range=0):
    X = df[pd.date_range(pred_start+timedelta(days=timesteps), pred_start+timedelta(days=1))].values
    if is_train: y = df[pd.date_range(pred_start, periods=16)].values
    else: y = None
    return X, y

timesteps = 100
# preparing data
train_data = train_generator(df, promo_df, items, stores, timesteps, date(2017, 1, 1),
                                           n_range=3, day_skip=1, batch_size=200, aux_as_tensor=False, reshape_output=2)

nextOne = next(train_data)









