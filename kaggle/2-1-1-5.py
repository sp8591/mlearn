# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
#zhfont = matplotlib.font_manager.FontProperties(fname="/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf") #字体
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

data = pd.read_csv('../Datasets/Titanic/titanic.txt')

print data.head()

x = data[['pclass', 'age', 'sex']]
y = data[['survived']]

print x.info()
print x
x['age'].fillna(x['age'].mean(), inplace=True)
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=33
)
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse=False)
print x_train.to_dict(orient='record')
x_train = vec.fit_transform(x_train.to_dict(orient='record'))
x_test = vec.transform(x_test.to_dict(orient='record'))

print x_train
print vec.feature_names_
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
y_predict = dtc.predict(x_test)

print dtc.score(x_test, y_test)
print classification_report(y_predict, y_test, target_names=['dead', 'survived'])
