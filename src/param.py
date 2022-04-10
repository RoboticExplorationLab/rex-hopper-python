"""
Copyright (C) 2021 Benjamin Bokser
"""
import numpy as np
import leg
import wbc


design_rw = {
    "model": "design_rw",
    "controllerclass": wbc,
    "legclass": leg,
    "csvpath": "res/hopper_rev06/urdf/hopper_rev06.csv",
    "urdfpath": "res/hopper_rev06/urdf/hopper_rev06.urdf",
    "init_q": [-30 * np.pi / 180, -120 * np.pi / 180, -150 * np.pi / 180, 120 * np.pi / 180],
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "hconst": 0.27,
    "n_a": 5,
    "ks": 996,
    "springpolarity": 1,
    "k": 10000,  # wbc gain
    "k_k": [45, 45*0.02],  # inv kin gain
}


design_cmg = {
    "model": "design_cmg",
    "controllerclass": wbc,
    "legclass": leg,
    "csvpath": "res/hopper_cmg_01/urdf/hopper_cmg_01.csv",
    "urdfpath": "res/hopper_cmg_01/urdf/hopper_cmg_01.urdf",
    "init_q": [-30 * np.pi / 180, -120 * np.pi / 180, -150 * np.pi / 180, 120 * np.pi / 180],
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "hconst": 0.27,
    "n_a": 9,
    "ks": 996,
    "springpolarity": 1,
    "k": 5000,  # wbc gain
    "k_k": [45, 45*0.02],  # inv kin gain  1
}

