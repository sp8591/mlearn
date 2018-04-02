
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

xx = pd.Series(np.exp(np.arange(20)))
xx.plot(label = 'dda', legend = True)
plt.show()
xx.plot(label = 'dd', logy=True, legend = True)
plt.show()