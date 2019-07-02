#from https://cloud.tencent.com/developer/article/1041442

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
 
# 转换序列成监督学习问题
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
 
# 加载数据集
dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# 整数编码
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# 归一化特征
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# 构建监督学习问题
reframed = series_to_supervised(scaled, 1, 1)
# 丢弃我们并不想预测的列
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)

print("reframed top5" , reframed[:5])
 
# 分割为训练集和测试集
values = reframed.values

print("reframed values", values[:5,:])
print("reframed values get values[:,:-1]", (values[:,:-1])[:5])
print("reframed values values[:,-1] ", (values[:,-1])[:5])

n_train_hours = 365 * 24
#from row beginning to n_train_hours, : means all columns
train = values[:n_train_hours, :]
#from row n_train_hours to the end, : means all columns
test = values[n_train_hours:, :]

# 分为输入输出
#train[:, :-1] => get all rows, column from beginning except the last one.
#train[:, -1]  => get all rows, columns only the last one.
train_X, train_y = train[:, :-1], train[:, -1]
print("Train_X top 5", train_X[:5])
print("Train_y top 5", train_y[:5])

test_X, test_y = test[:, :-1], test[:, -1]
# 重塑成3D形状 [样例, 时间步, 特征]
#shape to 3D, sampleCount, timeStep, features.
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print("train all shapes trainX, trainY, testX, testy ", train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print("train_X after reshape top 5 ", train_X[:5, :, :])
 
#params 
epochs=60
batch_size = 144
hiddenLayer = 100
#output one value
output = 1

# 设计网络
model = Sequential()
#LSTM or GRU both work
model.add(GRU(hiddenLayer, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(output))
model.compile(loss='mae', optimizer='adam')

# 拟合神经网络模型
history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# 绘制历史数据
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
 
# 做出预测
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# 反向转换预测值比例
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# 反向转换实际值比例
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# 计算RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
