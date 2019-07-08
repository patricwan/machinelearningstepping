
# coding: utf-8

# In[ ]:

#https://blog.csdn.net/slx_share/article/details/85098290
from statsmodels.tsa.arima_model import ARIMA 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
import matplotlib.pyplot as plt 
import pandas as pd 

df = pd.read_csv('international-airline-passengers.csv', skipfooter=0) 
df.columns = ['time', 'data'] 
df.dropna(inplace=True) 

print("original DF")
print(df.head())

df.index = pd.date_range(start='1949-01-31', end='1960-12-31', freq='M', ) 


y = df.data 
y.shape

# In[ ]:
print("after get df.data as y")
print(y.head())

model = SARIMAX(y[:120], order=(2, 2, 1), seasonal_order=(2, 1, 1, 12), freq='M').fit()
# print(model.summary()) 
y_pred = model.predict(start='1949-03-31', end='1960-12-31') 

plt.plot(y) 
plt.plot(y_pred) 

plt.legend(['y_true', 'y_pred']) 

plt.show()

