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

from sklearn.datasets import load_boston
boston = load_boston()
print boston.DESCR
x = boston.data
y = boston.target
y = np.reshape(y, [len(y), 1])
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.25,
    random_state=33
)
ss_x = StandardScaler()
ss_y = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.transform(y_test)

from sklearn.svm import SVR
linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train)
linear_svr_y_predict = linear_svr.predict(x_test)

poly_svr = SVR(kernel='poly')
poly_svr.fit(x_train, y_train)
poly_svr_y_predict = poly_svr.predict(x_test)

rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)


from sklearn.neighbors import KNeighborsRegressor
uni_knr = KNeighborsRegressor(weights='uniform')
uni_knr.fit(x_train, y_train)
uni_knr_y_predict = uni_knr.predict(x_test)

dis_knr = KNeighborsRegressor(weights='distance')
dis_knr.fit(x_train, y_train)
dis_knr_y_predict = dis_knr.predict(x_test)


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print r2_score(y_test, linear_svr_y_predict)

print mean_squared_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(linear_svr_y_predict)
)
print mean_absolute_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(linear_svr_y_predict)
)

print r2_score(y_test, poly_svr_y_predict)

print mean_squared_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(poly_svr_y_predict)
)
print mean_absolute_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(poly_svr_y_predict)
)

print r2_score(y_test, rbf_svr_y_predict)

print mean_squared_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(rbf_svr_y_predict)
)
print mean_absolute_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(rbf_svr_y_predict)
)

print r2_score(y_test, uni_knr_y_predict)

print mean_squared_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(uni_knr_y_predict)
)
print mean_absolute_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(uni_knr_y_predict)
)

print r2_score(y_test, dis_knr_y_predict)

print mean_squared_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(dis_knr_y_predict)
)
print mean_absolute_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(dis_knr_y_predict)
)

from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
dtr_y_predict = dtr.predict(x_test)

print r2_score(y_test, dtr_y_predict)

print mean_squared_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(dtr_y_predict)
)
print mean_absolute_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(dtr_y_predict)
)

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_y_predict = rfr.predict(x_test)

etr = ExtraTreesRegressor()
etr.fit(x_train, y_train)
etr_y_predict = etr.predict(x_test)

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
gbr_y_predict = gbr.predict(x_test)


print r2_score(y_test, rfr_y_predict)

print mean_squared_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(rfr_y_predict)
)
print mean_absolute_error(
    ss_y.inverse_transform(y_test),
    ss_y.inverse_transform(rfr_y_predict)
)

