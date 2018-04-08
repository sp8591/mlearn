# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

x = []
y = []
for i in range(4, 13):
    print '处理数据包%d' % i
    data = pd.read_csv('./data/package%d.csv' % i)
    print '读取数据%d条' % data.shape[0]
    data_not_null = data[data['投标价格'].notnull()]
    print '去除空数据%d条' % (data.shape[0] - data_not_null.shape[0])
    print '分析数据%d条' % (data_not_null.shape[0])
    X = 0.85
    Y = 1.2
    A1 = data_not_null['投标价格'].mean()
    A2_min = A1*X
    A2_max = A1*Y

    A2_data = data[(data['投标价格'] >= A2_min) & (data['投标价格'] <= A2_max)]

    A2 = A2_data['投标价格'].mean()

    a = 0.005
    print '使用如下指标:\n区间低配比X=%f\n区间高配比Y=%f\n下调a=%f\n' \
          '全部报价均值A1=%f\n调整后区间A2=[%f, %f]\n' \
          '区间报价均值A2=%f\n' \
          '区间内报价%d条' % \
          (X, Y, a, A1, A2_min, A2_max, A2, len(A2_data))
    tip = '使用如下计算公式：B = A2 * (1 - a)' if len(A2_data) >0 else '使用如下计算公式：B = A1 * (1 - a)'
    print tip
    B = A2 * (1 - a) if  len(A2_data) >0 else A1 * (1 - a)
    print '基准价B=%f' % (B)
    data_not_null = data_not_null.copy()
    data_not_null['基准价'] = B
    data_not_null['差值'] = data_not_null['投标价格'] - B
    data_not_null['差值绝对值'] = abs(data_not_null['投标价格'] - B)
    data_not_null['差值百分比%'] = (data_not_null['投标价格'] - B) / data_not_null['投标价格'] * 100

    data_not_null_sorted = data_not_null.sort_values(by='差值绝对值')
    data_not_null_sorted = data_not_null_sorted.reset_index(drop=True)
    del data_not_null_sorted['开标备注']
    print 'TOP 3:'
    print data_not_null_sorted.iloc[0:3, :]
    data_my = data_not_null_sorted[data_not_null_sorted['投标人名称'] == '山东硅钢新材料有限公司']
    my_index =  data_my.index.values
    print '硅钢公司排名:%d:' % my_index
    print data_my

    x.append(i)
    y.append(data_my['差值百分比%'].values[0])

    name = '包-%d' % i
    txt = '投标名称:%s\n' \
          '基准价:%.2f\n' \
          '投标价格:%.2f\n' \
          '差值:%.2f\n' \
          '差值百分比:%.2f%%\n' \
          '排名:%d' % \
          (name, data_my['基准价'], data_my['投标价格'],
           data_my['差值'], data_my['差值百分比%'], my_index)
    plt.annotate(txt, (x[-1], y[-1]), (x[-1] - 0.05, y[-1] + 0.003))
    print '-----------------------------------------------------------------------'
plt.plot(x, y, 'ro-', label='投标基准差百分比%')
plt.xlabel('包-')
plt.ylabel('投标基准差百分比%')
plt.legend()
plt.show()












