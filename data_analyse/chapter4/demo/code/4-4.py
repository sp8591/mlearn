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
inputfile = '../data/electricity_data.xls'
outputfile = '../tmp/electricity_data.xls'
data = pd.read_excel(inputfile)
data[u'线损率'] = (data[u'供入电量'] - data[u'供出电量'])/data[u'供入电量']
data.to_excel(outputfile, index = False) #保存结果

