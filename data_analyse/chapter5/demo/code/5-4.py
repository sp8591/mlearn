# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.interpolate import lagrange
from sklearn.manifold import TSNE
reload(sys)
sys.setdefaultencoding('utf-8')
#zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") #字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

inputfile = '../data/consumption_data.xls'
outputfile = '../tmp/data_type.xls'
outputfile_data = '../tmp/data_out.xls'
k = 3
iteration = 500
data = pd.read_excel(inputfile, index_col='Id')
data_zs = 1.0 * (data - data.mean()) / data.std()
from sklearn.cluster import KMeans
model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
model.fit(data_zs)

print data
print data_zs
print model.labels_
print model.cluster_centers_
r1 = pd.Series(model.labels_).value_counts()
r2 = pd.DataFrame(model.cluster_centers_)
r = pd.concat([r2, r1], axis=1)
print r
r.columns = list(data.columns) + [u'kind']
print r
r.to_excel(outputfile)

r1 = pd.Series(model.labels_)
#r = pd.concat([data, r1], axis=1)
data['kind'] = r1
print data

tsne = TSNE()
tsne.fit_transform(data_zs)
tsne = pd.DataFrame(tsne.embedding_, index=data_zs.index)
print data[u'kind']

d = tsne[data[u'kind'] == 0.0]

plt.plot(d[0], d[1], 'r.')

d = tsne[data[u'kind'] == 1.0]
plt.plot(d[0], d[1], 'go')

d = tsne[data[u'kind'] == 2.0]
plt.plot(d[0], d[1], 'b*')
plt.show()

def density_plots(data, title):
    print data
    for i in range(len(data.iloc[0])):
        print (data.iloc[:, i])
        (data.iloc[:, i]).plot(kind='kde', subplots=True, label=data.columns[i], linewidth=2, sharex=False)
        plt.show()
    plt.ylabel('dense')
    plt.xlabel('num1')
    plt.legend()

def density_plot(data):
    print data
    #plt.figure()
    p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
    plt.legend()
    plt.show()

pic_output = '.../tmp/pd_'
for i in range(k):
    #density_plot(data[data[u'kind'] == i].iloc[:, :3])
    density_plots(data[data[u'kind'] == i].iloc[:, :3], 'i')
    print r[u'kind'] == i

