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
    "aname": ["q0", "q2", "rw1", "rw2", "rw3"],  # actuator names
    "a_kt": np.array([1.73, 1.73, 0.106, 0.106, 0.0868]),  # actuator kt, including gear ratio
    "inertia": np.array([[0.07542817, 0.00016327,  0.00222099],
                         [0.00016327, 0.04599064,  -0.00008321],
                         [0.00222099, -0.00008321, 0.07709692]]),
    "rh": -np.array([0.02663114, 0.04435752, 6.61082088]) / 1000,  # mm to m
    "S": np.array([[1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1]]),  # actuator selection matrix
    "hconst": 0.27,  # default height
    "n_a": 5,  # number of actuators
    "ks": 3000,  # spring constant, N/m
    "springpolarity": 1,  # spring polarity
    "k": 5000,  # wbc gain
    "k_k": [45, 45*0.02],  # inv kin gain
    "mu": 2  # friction coeff at foot
}

design_rw_alt = {
    "model": "design_rw_alt",
    "controllerclass": wbc,
    "legclass": leg,
    "csvpath": "res/hopper_rev08/urdf/hopper_rev08_old.csv",
    "urdfpath": "res/hopper_rev08/urdf/hopper_rev08_old.urdf",
    "init_q": [-30 * np.pi / 180, -120 * np.pi / 180, -150 * np.pi / 180, 120 * np.pi / 180],
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "aname": ["q0", "q2", "rw1", "rw2", "rw3"],  # actuator names
    "a_kt": [0.247, 0.247, 0.106, 0.106, 0.0868],  # actuator kt
    "inertia": np.array([[0.07542817, 0.00016327,  0.00222099],
                         [0.00016327, 0.04599064,  -0.00008321],
                         [0.00222099, -0.00008321, 0.07709692]]),
    "rh": -np.array([0.02663114, 0.04435752, 6.61082088]) / 1000,  # mm to m
    "S": np.array([[1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 1]]),  # actuator selection matrix
    "hconst": 0.27,  # default height
    "n_a": 5,  # number of actuators
    "ks": 3000,  # spring constant
    "springpolarity": 1,  # spring polarity
    "k": 5000,  # wbc gain
    "k_k": [45, 45*0.02],  # inv kin gain
    "mu": 2  # friction coeff at foot
}

