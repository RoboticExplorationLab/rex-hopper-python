"""
Copyright (C) 2020 Benjamin Bokser
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from robotrunner import Runner


dt = 1e-3

parser = argparse.ArgumentParser()
parser.add_argument("--spring", help="add spring: True or False", action="store_true")
args = parser.parse_args()

if args.spring:
    spring = True
else:
    spring = False

i = 0
range = np.arange(0.3, 1.5, 0.1).reshape(-1, 1)
ft_mean = np.zeros(len(range))
for scale in range:
    print("Now running with scale of ", scale)
    runner = Runner(dt=dt, plot=False, model='design', ctrl_type='simple_invkin', fixed=False,
                    spring=spring, record=False, altsize=1, scale=scale, direct=True, total_run=10000)
    ft = runner.run()
    flighttimes = ft[ft != 0]
    ft_mean[i] = np.mean(flighttimes)
    i += 1

plt.scatter(range, ft_mean, label="Data")

poly = PolynomialFeatures(degree=2)
x = poly.fit_transform(range)
y = ft_mean.reshape(-1, 1)
clf = linear_model.LinearRegression().fit(x, y)
print("Best scale value = ", x[np.argmax(clf.predict(x))][1])
plt.plot(range, clf.predict(x), label="2nd Order Poly Fit", color='r')
plt.title('Flight Time vs Scale')
plt.xlabel("Scale")
plt.ylabel("Mean Flight Time, seconds")
plt.legend()
plt.show()