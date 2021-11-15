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

design = {
    "model": "design",
    "controllerclass": wbc_parallel,
    "legclass": leg_parallel,
    "csvpath": "res/flyhopper_robot/urdf/flyhopper_robot.csv",
    "urdfpath": "res/flyhopper_robot/urdf/flyhopper_robot.urdf",
    "linklengths": [.1, .3, .3, .1, .2, .0205],
    "k_kin": 37.5,
    "springpolarity": 1
}

i = 0
range = np.arange(0.3, 1.5, 0.1).reshape(-1, 1)
ft_mean = np.zeros(len(range))
for scale in range:
    print("Now running with scale of ", scale)
    runner = Runner(dt=dt, plot=False, model=design, ctrl_type='wbc_cycle', fixed=False,
                    spring=spring, record=False, scale=scale, direct=True, gravoff=False, total_run=10000)
    ft = runner.run()
    flighttimes = ft[ft != 0]
    ft_mean[i] = np.mean(flighttimes)
    i += 1

plt.scatter(range, ft_mean, label="Data")

poly = PolynomialFeatures(degree=3)
x = poly.fit_transform(range)
y = ft_mean.reshape(-1, 1)
clf = linear_model.LinearRegression().fit(x, y)
print("Best scale value = ", x[np.argmax(clf.predict(x))][1])
plt.plot(range, clf.predict(x), label="3rd Order Poly Fit", color='r')
plt.title('Flight Time vs Scale')
plt.xlabel("Scale")
plt.ylabel("Mean Flight Time, seconds")
plt.legend()
plt.show()