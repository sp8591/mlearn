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
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
path  = '../Datasets/Breast-Cancer/all.csv'
data = pd.read_csv(path, names=column_names)
data = data.replace(to_replace='?', value=np.nan)
data = data.dropna(how='any')
print data

x_train, x_test, y_train, y_test = train_test_split(
    data[column_names[1: 10]],
    data[column_names[10]],
    test_size=0.25,
    random_state=33,
)
print x_train, x_test, y_train, y_test
print x_train.describe()
print x_test.describe()

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

lr = LogisticRegression()
sgdc = SGDClassifier()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
sgdc.fit(x_train, y_train)
sgdc_y_predict = sgdc.predict(x_test)

from sklearn.metrics import classification_report
print lr.score(x_test, y_test)
print classification_report(y_test, lr_y_predict,
                            target_names=['Benign', 'Malignant'])
print y_test.value_counts()
print sgdc.score(x_test, y_test)
print classification_report(y_test, sgdc_y_predict,
                            target_names=['Benign', 'Malignant'])