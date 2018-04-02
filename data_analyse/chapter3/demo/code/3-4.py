# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") #字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

catering_safe  = '../data/catering_sale_all.xls'
data = pd.read_excel(catering_safe, index_col=u'日期')
print data.corr()
print data.corr()[u'百合酱蒸凤爪']
print data[u'百合酱蒸凤爪'].corr(data[u'翡翠蒸香茜饺'])
print data.ix[:, :]
print data.ix[[1, 2, 3], :]


data.plot(kind='hist', subplots=True)

plt.show()
