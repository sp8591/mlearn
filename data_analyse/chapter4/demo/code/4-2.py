
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

datafile = '../data/normalization_data.xls'
data = pd.read_excel(datafile, header=None)
print data
print (data - data.min())/(data.max() - data.min())
print (data - data.mean())/(data.std())
print data/10**np.ceil(np.log10(data.abs().max()))
