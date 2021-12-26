"""
Copyright (C) 2021 Benjamin Bokser
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

import param
import leg_parallel
import wbc_parallel
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
range1 = np.arange(0.5, 1.5, 0.1).reshape(-1, 1)
range2 = np.arange(3500, 5500, 400).reshape(-1, 1)
ft_mean = np.zeros(len(range1))
best_gain = np.zeros(len(range1))
for scale in range1:
    print("Now running with scale of ", scale)
    j = 0
    ftg_mean = np.zeros(len(range2)+1)
    for gain in range2:
        print("...using gain of ", gain)
        runner = Runner(dt=dt, plot=False, model=param.design, ctrl_type='wbc_cycle', fixed=False,
                        spring=spring, record=False, scale=scale, direct=True, gravoff=False, total_run=4000, gain=gain)
        ft = runner.run()
        flighttimes = ft[ft != 0]
        ftg_mean[j] = np.mean(flighttimes)
        j += 1
    ft_mean[i] = np.amax(ftg_mean)
    best_gain[i] = np.argmax(ftg_mean) + 1
    print("Best gain for scale of ", scale, " is ", best_gain[i])
    i += 1

matplotlib.rc('font', family='Times New Roman')
#font = {'fontname':'Times New Roman'}
plt.scatter(range1, ft_mean, label="Data")

poly = PolynomialFeatures(degree=3)
x = poly.fit_transform(range1)
y = ft_mean.reshape(-1, 1)
clf = linear_model.LinearRegression().fit(x, y)
print("Best scale value = ", x[np.argmax(clf.predict(x))][1])
print("Best gain values = ", best_gain)
plt.plot(range1, clf.predict(x), label="3rd Order Poly Fit", color='r')
# plt.title('Flight Time vs Scale')
plt.xlabel("Scale")
plt.ylabel("Mean Flight Time, seconds")
plt.legend()
plt.show()