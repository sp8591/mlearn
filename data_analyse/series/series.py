# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from pandas import Series
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") #字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

file = './car_sales.csv'

series = pd.read_csv(file, header=0, index_col='Month')
# print series['Monthly car sales in Quebec 1960-1968']['1960-01']
# series.plot()
# plt.show()
#
series.index = pd.to_datetime(series.index)

#for differ
differenced = series.diff(12,)
differenced = differenced[12:]
differenced.to_csv('seasonally_adj.csv')
#print series['Monthly car sales in Quebec 1960-1968']['1960']

#for roll_mean
size = 2
timeSeries = series['Monthly car sales in Quebec 1960-1968']
f = plt.figure()
rol_mean = timeSeries.rolling(window=size).mean()
rol_weighted_mean = pd.ewma(timeSeries, span=size)
timeSeries.plot(color='blue', label='original')
rol_mean.plot(color='red', label='rol_mean')
rol_weighted_mean.plot(color='green', label='rol_weighted_mean')
plt.legend(loc='best')
plt.title('roll mean')
plt.show()

#for autocorr
from statsmodels.graphics.tsaplots import  plot_acf
plot_acf(timeSeries)
plt.show()