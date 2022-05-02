"""
Copyright (C) 2021-2022 Benjamin Bokser
"""
import numpy as np
import leg
import wbc


design_rw = {
    "model": "design_rw",
    "controllerclass": wbc,
    "legclass": leg,
    "csvpath": "res/hopper_rev08/urdf/hopper_rev08.csv",
    "urdfpath": "res/hopper_rev08/urdf/hopper_rev08.urdf",
    "init_q": [-30 * np.pi / 180, -120 * np.pi / 180, -150 * np.pi / 180, 120 * np.pi / 180],
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "aname": ["q0", "q2", "rw1", "rw2", "rw3"],
    "inertia": np.array([[76148072.89, 70089.52,    2067970.36],
                         [70089.52,    45477183.53, -87045.58],
                         [2067970.36,  -87045.58,   76287220.47]]) * (10 ** (-9)),
    "rh": np.array([0., 0., 0.]),  # np.array([0.02201854, 6.80044366, 0.97499173]) / 1000,  # mm to m
    "S": np.array([[1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1]]),
    "hconst": 0.27,  # height constant
    "n_a": 5,  # number of actuators
    "ks": 996,  # spring constant
    "springpolarity": 1,  # spring polarity
    "k": 5000,  # wbc gain
    "k_k": [45, 45*0.02],  # inv kin gain
    "mu": 0.5  # friction coeff at foot
}

