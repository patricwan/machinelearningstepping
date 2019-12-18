from datetime import date, timedelta
import pandas as pd
import numpy as np
from pandas import Series, DataFrame


class DataLoader():
    def __init__(self, csvPath):
        self.csvPath = csvPath
        self.loadData()

    def loadData(self):
        self.df_train = pd.read_csv(self.csvPath)
        self.df_train["everyDayHour"] = self.df_train['everyDayHour'].map(lambda strDateHour: "20" + strDateHour[6:8]
                                                                                              + "/" + strDateHour[
                                                                                                      0:5] + " "
                                                                                              + strDateHour[
                                                                                                9:11] + ":00:00")
        self.df_train["metric"] = "cpu"
        self.df_train["everyDayHour"] = pd.to_datetime(self.df_train["everyDayHour"], format='%Y%m%d %H:%M:%S')
        self.df_train = self.df_train.astype({"avg(cpu)": 'float'})

        self.df_train = self.df_train.pivot(index="metric", columns="everyDayHour", values="avg(cpu)")

        print(self.df_train.head())

    def get_timespan(self, df, datetimeStart, minus, periods, freq='H'):
        return df[pd.date_range(datetimeStart - timedelta(hours=minus), periods=periods, freq=freq)]

    def prepare_dataset(self, df, datetimeStart, is_train=True, name_prefix=None):
        X = {}
        # for i in [2, 7, 14, 28]:
        for i in [2]:
            tmp = self.get_timespan(df, datetimeStart, i, i)
            # X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
            # X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
            # X['mean_%s' % i] = tmp.mean(axis=1).values
            X['median_%s' % i] = tmp.median(axis=1).values
            # X['min_%s' % i] = tmp.min(axis=1).values
            # X['max_%s' % i] = tmp.max(axis=1).values
            # X['std_%s' % i] = tmp.std(axis=1).values

        for i in range(1, 24):
            X['hour_%s_2019' % i] = self.get_timespan(df, datetimeStart, i, 1).values.ravel()

        X = pd.DataFrame(X)

        # start date and periods is 24 hours
        if is_train:
            yInput = df[pd.date_range(datetimeStart, periods=24, freq='H')].values
            yTarget = df[pd.date_range(datetimeStart + timedelta(hours=1), periods=24, freq='H')].values
            return X, yInput, yTarget

        return X

    # startDateTime = "2019-05-21 02:00:00"
    # numHours = 3800
    def prepareTrainingData(self, startDateTime, numHours):
        X_l, y_lIn, y_lTg = [], [], []

        t2019Jul19 = pd.to_datetime(startDateTime, format='%Y%m%d %H:%M:%S')
        num_hours = numHours
        for i in range(num_hours):
            delta = timedelta(hours=i)
            X_tmp, y_tmp1, y_tmp2 = self.prepare_dataset(self.df_train, t2019Jul19 + delta)

            X_l.append(X_tmp)
            y_lIn.append(y_tmp1)
            y_lTg.append(y_tmp2)

        X_train = pd.concat(X_l, axis=0)
        print(X_train.shape)
        # DataFrame(X_train).to_csv("x_train_sfdc.csv")

        y_trainIn = np.concatenate(y_lIn, axis=0)
        y_trainTarget = np.concatenate(y_lTg, axis=0)
        # DataFrame(y_trainIn).to_csv("y_trainIn_sfdc.csv")
        # DataFrame(y_trainTarget).to_csv("y_trainTarget_sfdc.csv")
        X_train = X_train.as_matrix()

        # reshape to 3D shape for X
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        print(X_train.shape)

        y_trainIn = DataFrame(y_trainIn).as_matrix()
        y_trainIn = y_trainIn.reshape((y_trainIn.shape[0], y_trainIn.shape[1], 1))
        print(y_trainIn.shape)

        y_trainTarget = DataFrame(y_trainTarget).as_matrix()
        y_trainTarget = y_trainTarget.reshape((y_trainTarget.shape[0], y_trainTarget.shape[1], 1))

        return X_train, y_trainTarget

    def generate_x_y_data(self, X_train, y_trainTarget, isTrain=True, batch_size=10):
        # shape: (batch_size, seq_length, output_dim)
        idx = 0
        while True:
            if idx + batch_size > X_train.shape[0]:
                idx = 0
            start = idx
            idx += batch_size
            yield X_train[start:start + batch_size, :, :], y_trainTarget[start:start + batch_size, :, :]

        # shape: (seq_length, batch_size, output_dim)
        return None


dataLoader = DataLoader("../../../../data/sflogs/loadCPU04t.csv")
X_train, y_trainTarget = dataLoader.prepareTrainingData("2019-05-21 02:00:00", 3800)


