# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") #字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0)
plt.figure()
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.axis('equal')

plt.figure()
x = np.random.randn(1000)
plt.hist(x, 10)


x = np.random.randn(1000)
D = pd.DataFrame([x, x+1]).T
D.plot(kind= 'box')

xx = pd.Series(np.exp(np.arange(20)))
xx.plot(label = 'dd', legend = True)
plt.show()
xx.plot(label = 'dd', logy=True, legend = True)


plt.show()

