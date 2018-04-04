# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") #字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
from sklearn.datasets import load_digits
digits = load_digits()
print digits.data.shape
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=33
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
y_predict = lsvc.predict(x_test)
lsvc.score(x_test, y_test)
from sklearn.metrics import classification_report
print classification_report(y_test, y_predict, labels=digits.target_names.astype(str))