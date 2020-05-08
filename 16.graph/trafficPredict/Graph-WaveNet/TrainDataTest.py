import torch
import numpy as np
import argparse
import time
import os
import pandas as pd

#python generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

if __name__ == "__main__":
    print("This is the start of train data test main program")
    seq_length_x = 12
    seq_length_y = 12
    df = pd.read_hdf("/root/github/data/traffic/metr-la.h5")

    print("DF columns ", df.columns.values)

    y_start  = 1

    # 0 is the latest observed sample.
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    # Predict the next one hour
    y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))

    # x: (num_samples, input_length, num_nodes, input_dim)
    # y: (num_samples, output_length, num_nodes, output_dim)
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]

    add_time_in_day = True

    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    print("x sample " , x[0])
    print("y sample " , y[0])
    #x = np.stack(x, axis=0)
    #y = np.stack(y, axis=0)















