# coding:utf-8
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import sys
from scipy.interpolate import lagrange
reload(sys)
sys.setdefaultencoding('utf-8')
#zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") #字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

inputfile = '../data/principal_component.xls'
outputfile = '../tmp/dimention_reduced.xls'
data = pd.read_excel(inputfile, header=None)
print data
print data[0:3][0:2]
from sklearn.decomposition import PCA
pca = PCA(3)
print pca.fit(data)
data = asarray(data)

print pca.components_
print pca.explained_variance_ratio_
low_d = pca.transform(data)
pd.DataFrame(low_d).to_excel(outputfile)
