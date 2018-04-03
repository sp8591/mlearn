# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.interpolate import lagrange
reload(sys)
sys.setdefaultencoding('utf-8')
#zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") #字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

datafile = '../data/sales_data.xls'
data = pd.read_excel(datafile, index_col = u'序号')
print data.columns
print data
data[data == u'好'] = 1
data[data == u'是'] = 1
data[data == u'高'] = 1
data[data != 1] = -1
x = data.iloc[:, :3].as_matrix().astype(int)
y = data.iloc[:, 3].as_matrix().astype(int)
from sklearn.tree import DecisionTreeClassifier as DTC
dtc = DTC(criterion='entropy')
dtc.fit(x, y)
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
with open('tree.dot', 'w') as f:
    f = export_graphviz(dtc, feature_names=data.iloc[:, :3].columns, out_file=f)

