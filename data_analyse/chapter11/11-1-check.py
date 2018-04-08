# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

datafile = './discdata.xls'
transformeddata = './discdata_processed.xls'
data = pd.read_excel(transformeddata)

data = data.iloc[: len(data) - 5]

from statsmodels.tsa.stattools import  adfuller as ADF
diff = 0
adf = ADF(data['CWXT_DB:184:D:\\'])
while adf[1] >= 0.05:
  diff = diff + 1
  adf = ADF(data['CWXT_DB:184:D:\\'].diff(diff).dropna())

print 's %s, p %s' % (diff, adf[1])


from statsmodels.stats.diagnostic import acorr_ljungbox

[[lb], [p]] = acorr_ljungbox(data['CWXT_DB:184:D:\\'], lags=1)
print p

[[lb], [p]] = acorr_ljungbox(data['CWXT_DB:184:D:\\'].diff().dropna(), lags=1)
print p

