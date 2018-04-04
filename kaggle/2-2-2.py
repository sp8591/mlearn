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


import numpy as np
M = np.array([[1, 2], [2, 4]])
print np.linalg.matrix_rank(M, )

# 导入pandas用于数据读取和处理。
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split

# # 从互联网读入手写体图片识别任务的训练数据，存储在变量digits_train中。
# digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra', header=None)

# # 从互联网读入手写体图片识别任务的测试数据，存储在变量digits_test中。
# digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes', header=None)

# # 分割训练数据的特征向量和标记。
# X_digits = digits_train[np.arange(64)]
# y_digits = digits_train[64]

data = load_digits()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=33, test_size=0.25)

# 从sklearn.decomposition导入PCA。
from sklearn.decomposition import PCA

# 初始化一个可以将高维度特征向量（64维）压缩至2个维度的PCA。
estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_train)

# 显示10类手写体数字图片经PCA压缩后的2维空间分布。
from matplotlib import pyplot as plt


def plot_pca_scatter():
    colorss = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'gray']
    for i in xrange(len(colorss)):
        px = X_pca[:, 0][y_train == i]
        py = X_pca[:, 1][y_train == i]
        plt.scatter(px, py, c=colorss[i])

    plt.legend(np.arange(2, 12).astype(str))
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()

plot_pca_scatter()
nn = np.array([[1, 2], [2, 4]])



# 导入基于线性核的支持向量机分类器。
from sklearn.svm import LinearSVC

# 使用默认配置初始化LinearSVC，对原始64维像素特征的训练数据进行建模，并在测试数据上做出预测，存储在y_predict中。
svc = LinearSVC()
svc.fit(X_train, y_train)
y_predict = svc.predict(X_test)

# 使用PCA将原64维的图像数据压缩到20个维度。
estimator = PCA(n_components=20)

# 利用训练特征决定（fit）20个正交维度的方向，并转化（transform）原训练特征。
pca_X_train = estimator.fit_transform(X_train)
# 测试特征也按照上述的20个正交维度方向进行转化（transform）。
pca_X_test = estimator.transform(X_test)

# 使用默认配置初始化LinearSVC，对压缩过后的20维特征的训练数据进行建模，并在测试数据上做出预测，存储在pca_y_predict中。
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_y_predict = pca_svc.predict(pca_X_test)
