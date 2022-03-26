"""
Copyright (C) 2021 Benjamin Bokser
"""
import leg_parallel
import wbc_parallel
import gait

design_cmg = {
    "model": "design_cmg",
    "controllerclass": wbc_parallel,
    "legclass": leg_parallel,
    "csvpath": "res/hopper_cmg_01/urdf/hopper_cmg_01.csv",
    "urdfpath": "res/hopper_cmg_01/urdf/hopper_cmg_01.urdf",
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "k": 5000,  # wbc gain
    "k_k": [45, 45*0.02],  # inv kin gain  1
    "springpolarity": 1,
    "hconst": 0.27,
    "n_a": 9
}

design_rw = {
    "model": "design_rw",
    "controllerclass": wbc_parallel,
    "legclass": leg_parallel,
    "csvpath": "res/hopper_rev06/urdf/hopper_rev06.csv",
    "urdfpath": "res/hopper_rev06/urdf/hopper_rev06.urdf",
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "k": 5000,  # wbc gain
    "k_k": [45, 45*0.02],  # inv kin gain
    "springpolarity": 1,
    "hconst": 0.27,
    "n_a": 5
}