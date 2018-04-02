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

datafile = '../data/discretization_data.xls'
data = pd.read_excel(datafile)
print data

data = data[u'肝气郁结证型系数'].copy()

print data
k = 4
d1 = pd.cut(data, k, labels = range(k))
print d1

w = [1.0 * i/k for i in range(k+1)]
ww = data.describe(percentiles = w)
print ww
w = ww[4: 4+k+1]
print w
print w[0]
#w[0] = w[0]*(1-1e-10)
d2 = pd.cut(data, w, labels=range(k))
from sklearn.cluster import KMeans
kmodel = KMeans(n_clusters=k, n_jobs=4)
da = data.reshape(len(data), 1)
kmodel.fit(da)
c = pd.DataFrame(kmodel.cluster_centers_).sort_values(0)
print c
ww = pd.rolling_mean(c, 2)
print ww
w = ww.iloc[1:]

print w
w = [0] + list(w[0]) + [data.max()]
d3 = pd.cut(data, w, labels=range(k))
def cluster_plot(d,k):
    plt.figure()
    for j in range(0, k):
        plt.plot(data[d==j], [j for i in d[d==j]], 'o')
    plt.ylim(-0.5, k - 0.5)

cluster_plot(d1, k)
cluster_plot(d2, k)
cluster_plot(d3, k)
plt.show()