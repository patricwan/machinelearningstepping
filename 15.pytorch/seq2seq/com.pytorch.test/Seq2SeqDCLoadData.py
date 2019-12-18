from datetime import date, timedelta
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

class DCLoadDataLoader():
    def __init__(self, csvPath):
        self.csvPath = csvPath
        self.loadData()

    def loadData(self):
        self.df_train = pd.read_csv(self.csvPath)
        self.df_train["everyDayHour"] = self.df_train['everyDayHour'].map(lambda strDateHour: "20" + strDateHour[6:8]
                                                                                              + "/" + strDateHour[0:5] + " "
                                                                                              + strDateHour[9:11] + ":00:00")
        self.df_train["everyDayHour"] = pd.to_datetime(self.df_train["everyDayHour"], format='%Y%m%d %H:%M:%S')
        self.df_train = self.df_train.astype({"avgCPU": 'double'})
        self.df_train = self.df_train.astype({"avgMemUsed": 'double'})
        self.df_mapData = self.transformDataToMap(self.df_train)
        print(self.df_mapData)

        self.df_mapDateHourVector = {}

        self.df_train = self.df_train.set_index(["everyDayHour"])
        self.df_train = self.df_train.T
        #print(self.df_train.head())

    def transformDataToMap(self, dataSet):
        print(dataSet.head())
        allDayHours = dataSet["everyDayHour"]
        iCount = 0
        returnMap = {}
        for eachDateHour in allDayHours:
            eachRowValues = dataSet.iloc[iCount].values
            cpumemEach = [eachRowValues[1],eachRowValues[2]/1000]

            workDayHour = pd.to_datetime(eachDateHour, format='%Y%m%d %H:%M:%S')
            returnMap[workDayHour] = cpumemEach
            iCount = iCount + 1
        return returnMap

    def get_timespan(self, df, datetimeStart, minus, periods, freq='H'):
        return df[pd.date_range(datetimeStart - timedelta(hours=minus), periods=periods, freq=freq)]

    def prepare_dataset(self, df, datetimeStart, is_train=True, name_prefix=None):
        X = {}
        # for i in [2, 7, 14, 28]:
        for i in [6]:
            tmp = self.get_timespan(df, datetimeStart, i, i)
            #print("tmp timespan:\r\n" , tmp)
            X['diff_%s_mean' % i] = tmp.diff(axis=1).mean(axis=1).values
            X['mean_%s_decay' % i] = (tmp * np.power(0.9, np.arange(i)[::-1])).sum(axis=1).values
            X['mean_%s' % i] = tmp.mean(axis=1).values
            X['median_%s' % i] = tmp.median(axis=1).values
            X['min_%s' % i] = tmp.min(axis=1).values
            X['max_%s' % i] = tmp.max(axis=1).values
            X['std_%s' % i] = tmp.std(axis=1).values

        X = pd.DataFrame(X)

        # start date and periods is 24 hours
        if is_train:
            yInput = df[pd.date_range(datetimeStart, periods=24, freq='H')].values
            yTarget = df[pd.date_range(datetimeStart + timedelta(hours=1), periods=24, freq='H')].values
            return X, yInput, yTarget

        return X

    def prepareTrainingData(self, startDateTime, numHours):
        X_l, y_lIn, y_lTg = [], [], []

        t2019Jul19 = pd.to_datetime(startDateTime, format='%Y%m%d %H:%M:%S')
        num_hours = numHours
        for i in range(num_hours):
            delta = timedelta(hours=i)
            self.prepare_dataset(self.df_train, t2019Jul19 + delta)

    def completeAllVectors(self):
        for eachDayHour in self.df_mapData.keys():
            processedDateTime = eachDayHour + timedelta(hours=24)
            if processedDateTime in self.df_mapData:
                self.completeOneVector(processedDateTime)

    def completeOneVector(self, eachDayHour):
        oneVector = self.df_mapData[eachDayHour]
        workDayHour = pd.to_datetime(eachDayHour, format='%Y%m%d %H:%M:%S')

        rangedData = self.get_timespan(self.df_train, workDayHour, 24, 24)

        finalVector = np.append(oneVector, rangedData.diff(axis=1).mean(axis=1).values)
        finalVector = np.append(finalVector, rangedData.mean(axis=1).values)
        finalVector = np.append(finalVector, rangedData.median(axis=1).values)
        finalVector = np.append(finalVector, rangedData.min(axis=1).values)
        finalVector = np.append(finalVector, rangedData.max(axis=1).values)
        finalVector = np.append(finalVector, rangedData.std(axis=1).values)

        finalVector = np.append(finalVector,[workDayHour.hour, workDayHour.dayofweek])

        self.df_mapDateHourVector[eachDayHour] = finalVector


    #seq_length 24, data_size, input_dim(ouput_dim)
    def prepare3DData(self):
        self.completeAllVectors()

        finalMatrixX_input = []
        finalMatrixY_output = []
        for eachDate in self.df_mapDateHourVector.keys():
            if ( not (eachDate in self.df_mapDateHourVector) or not (eachDate + timedelta(hours=24) in self.df_mapDateHourVector)):
                continue
            oneSeqRes = self.df_mapDateHourVector[eachDate]
            outputSeqRes = self.df_mapDateHourVector[eachDate + timedelta(hours=24)]

            for indexDelta in range(1, 24):
                dateIntervalEnd = eachDate + timedelta(hours=indexDelta)
                dateOuputIntervalEnd = eachDate + timedelta(hours=indexDelta + 24)
                if (not (dateIntervalEnd in self.df_mapDateHourVector) or not (dateOuputIntervalEnd in self.df_mapDateHourVector)):
                    continue
                oneSeqRes = np.vstack((oneSeqRes, self.df_mapDateHourVector[dateIntervalEnd]))
                outputSeqRes = np.vstack((outputSeqRes, self.df_mapDateHourVector[dateOuputIntervalEnd]))

            oneSeqRes = np.array(oneSeqRes)
            outputSeqRes = np.array(outputSeqRes)
            if (oneSeqRes.shape[0] !=24 or outputSeqRes.shape[0] !=24):
                continue

            oneSeqRes = oneSeqRes.reshape((oneSeqRes.shape[0], oneSeqRes.shape[1],1))
            outputSeqRes = outputSeqRes.reshape((outputSeqRes.shape[0], outputSeqRes.shape[1], 1))
            if (len(finalMatrixX_input) == 0):
                finalMatrixX_input = oneSeqRes
                finalMatrixX_input = finalMatrixX_input.reshape((finalMatrixX_input.shape[0], finalMatrixX_input.shape[1], 1))

                finalMatrixY_output = outputSeqRes
                finalMatrixY_output = finalMatrixY_output.reshape((finalMatrixY_output.shape[0], finalMatrixY_output.shape[1], 1))
            else:
                finalMatrixX_input = np.concatenate((finalMatrixX_input, oneSeqRes), axis = 2)
                finalMatrixY_output = np.concatenate((finalMatrixY_output, outputSeqRes), axis=2)

        finalMatrixX_input = np.array(finalMatrixX_input)
        finalMatrixY_output = np.array(finalMatrixY_output)
        print("finalMatrixX_input shape" , finalMatrixX_input.shape)
        print("finalMatrixY_output shape", finalMatrixY_output.shape)

        return finalMatrixX_input, finalMatrixY_output[:,0:2,:]

def prepareData(csvPath):
    dataLoader = DCLoadDataLoader(csvPath)

    xInput, yOutput = dataLoader.prepare3DData()
    return xInput, yOutput

#dataLoader = DCLoadDataLoader("../../../../data/sflogs/DC4cfCPUMemData.csv")

#xInput, yOutput = dataLoader.prepare3DData()
#print("xShape ", xInput.shape)
#print("yShape ", yOutput.shape)


