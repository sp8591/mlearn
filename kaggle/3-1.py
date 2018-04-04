# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import sys
from sklearn.cluster import KMeans
from scipy.spatial.distance import  cdist
reload(sys)
sys.setdefaultencoding('utf-8')
#zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") #字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

data = np.array(['i am a teacher teacher!', 'i am an apple!'])
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()
data_count = count_vector.fit_transform(data)
print data, data_count, count_vector.vocabulary_