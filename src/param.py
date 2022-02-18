"""
Copyright (C) 2021 Benjamin Bokser
"""
import leg_parallel
import wbc_parallel


design_cmg = {
    "model": "design_cmg",
    "controllerclass": wbc_parallel,
    "legclass": leg_parallel,
    "csvpath": "res/hopper_cmg_01/urdf/hopper_cmg_01.csv",
    "urdfpath": "res/hopper_cmg_01/urdf/hopper_cmg_01.urdf",
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "k": 5000,  # wbc gain
    "k_g": 45,  # inv kin gain
    "k_gd": 45*0.02,
    "k_a": 1,
    "k_ad": 1*0.08,
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
    #"csvpath": "res/flyhopper_rwz/urdf/flyhopper_rwz.csv",
    #"urdfpath": "res/flyhopper_rwz/urdf/flyhopper_rwz.urdf",
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "k": 5000,  # wbc gain
    "k_g": 45,  # inv kin gain
    "k_gd": 45*0.02,
    "k_a": 1,
    "k_ad": 1*0.08,
    "springpolarity": 1,
    "hconst": 0.27,
    "n_a": 5
}