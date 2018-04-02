# coding:utf-8
import pandas as pd
dish_profit = '../data/catering_dish_profit.xls'
data = pd.read_excel(dish_profit, index_col = u'菜品名')
data = data[u'盈利'].copy()
data.sort(ascending = False)
import matplotlib.pyplot as plt #导入图像库
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
plt.figure()
data.plot(kind='bar')
plt.ylabel(u'盈利（元）')
p = 1.0 * data.cumsum()/data.sum()
p.plot(color = 'r', secondary_y = True, style = '-o', linewidth = 2)
