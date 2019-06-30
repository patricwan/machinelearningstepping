from pandas import read_csv
from datetime import datetime
# 加载数据
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')
dataset = read_csv('../../../data/air/PRSA_data_2010.1.1-2014.12.31.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
dataset.drop('No', axis=1, inplace=True)
# 手动更改列名
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'
# 把所有NA值用0替换
dataset['pollution'].fillna(0, inplace=True)
# 丢弃前24小时
dataset = dataset[24:] 
# 输出前五行
print(dataset.head(5))
# 保存到文件中
dataset.to_csv('pollution.csv')

from pandas import read_csv
from matplotlib import pyplot

dataset = read_csv('pollution.csv', header=0, index_col=0)
values = dataset.values
# 指定要绘制的列
groups = [0, 1, 2, 3, 5, 6, 7]
i = 1
# 绘制每一列
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()
