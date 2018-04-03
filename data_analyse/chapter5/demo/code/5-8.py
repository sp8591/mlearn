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

inputfile = '../data/consumption_data.xls'
k = 3
threshold = 2
iteration = 500
data = pd.read_excel(inputfile, index_col='Id')
data_zs = 1.0 * (data -data.mean())/ data.std()
from sklearn.cluster import KMeans
model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
model.fit(data_zs)
r = pd.concat([data_zs, pd.Series(model.labels_, index=data.index)], axis=1)
r.columns = list(data.columns) + [u'kind']
norm = []
for i in range(k):
    norm_tmp = r[['R', 'F', 'M']][r[u'kind'] == i] - model.cluster_centers_[i]
    norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)
    norm.append(norm_tmp/norm_tmp.median())

norm = pd.concat(norm)
print norm
discrete_point = norm[norm <= threshold]
discrete_point.plot(style = 'go')
discrete_points = norm[norm > threshold]
discrete_points.plot(style = 'ro')
for i in range(len(discrete_points)):
    id = discrete_points.index[i]
    n = discrete_points.iloc[i]
    plt.annotate('(%s, %0.2f)' % (id, n), xy=(id, n), xytext=(id, n))
plt.show()





