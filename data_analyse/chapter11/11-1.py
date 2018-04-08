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
data = pd.read_excel(datafile)


data = data[data['TARGET_ID'] == 184].copy()

data_group = data.groupby('COLLECTTIME')
print data_group
def attr_trans(x): #定义属性变换函数
  result = pd.Series(index = ['SYS_NAME', 'CWXT_DB:184:C:\\', 'CWXT_DB:184:D:\\',	'COLLECTTIME'])
  result['SYS_NAME'] = x['SYS_NAME'].iloc[0]
  result['COLLECTTIME'] = x['COLLECTTIME'].iloc[0]
  result['CWXT_DB:184:C:\\'] = x['VALUE'].iloc[0]
  result['CWXT_DB:184:D:\\'] = x['VALUE'].iloc[1]
  return result

data_processed = data_group.apply(attr_trans) #逐组处理
data_processed.to_excel(transformeddata, index = False)

