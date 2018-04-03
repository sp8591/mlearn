# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from sklearn.manifold import TSNE
from scipy.interpolate import lagrange
reload(sys)
sys.setdefaultencoding('utf-8')
#zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") #字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

filename = '../data/bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:, :8]
y = data.iloc[:, 8]
print x
print y
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
rlr = RLR()
rlr.fit(x, y)
a = rlr.get_support()
x = data[data.columns[rlr.get_support()]].as_matrix()
print ','.join(data.columns[a])
lr = LR()
lr.fit(x, y)
print lr.score(x, y)


