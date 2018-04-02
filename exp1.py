import numpy as np
print np.__version__
a = [1, 2, 3]
b = [2, 3, 4]
c = [i + 2 for i in a]
print c
c = map(lambda x: x + 2, a)
print c
d = map(lambda  x, y: x + y, a, b)
print d
e = reduce(lambda x, y: x + y, a)
print e

b = filter(lambda x: x > 5 and x < 8, range(10))
print b
import math
print math.sin(1)

print math.exp(1.0)

import pandas as pd
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
d = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns= ['a', 'b', 'c'])
d2 = pd.DataFrame(s)

print d.head()
print d.describe()

from scipy.optimize import fsolve
def f(x):
    x1 = x[0]
    x2 = x[1]
    return [2*x1 - x2**2 -1, x1**2 -x2 - 2]

result = fsolve(f, [1, 1])
print result

from scipy import integrate
def g(x):
    return (1 - x**2)**0.5
pi_2, err = integrate.quad(g, -1, 1)
print (pi_2 * 2)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
print model

from sklearn import datasets
iris = datasets.load_iris()
print iris.data.shape
from sklearn import svm
clf = svm.LinearSVC()
clf.fit(iris.data, iris.target)
print clf.predict([[5.0, 3.6, 1.3, 0.25]])
print clf.coef_
