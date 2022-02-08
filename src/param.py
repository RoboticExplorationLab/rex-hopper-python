"""
Copyright (C) 2021 Benjamin Bokser
"""
import leg_parallel
import wbc_parallel
import leg_serial
import wbc_serial
import leg_belt


design_cmg = {
    "model": "design_cmg",
    "controllerclass": wbc_parallel,
    "legclass": leg_parallel,
    "csvpath": "res/hopper_cmg/urdf/hopper_cmg.csv",
    "urdfpath": "res/hopper_cmg/urdf/hopper_cmg.urdf",
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "k": 5000,  # wbc gain
    "k_g": 45,  # inv kin gain
    "k_gd": 45*0.02,
    "k_a": 1,
    "k_ad": 1*0.08,
    "springpolarity": 1,
    "hconst": 0.27
}

design_rw = {
    "model": "design_rw",
    "controllerclass": wbc_parallel,
    "legclass": leg_parallel,
    "csvpath": "res/hopper_rev05/urdf/hopper_rev05.csv",
    "urdfpath": "res/hopper_rev05/urdf/hopper_rev05.urdf",
    #"csvpath": "res/flyhopper_rwz/urdf/flyhopper_rwz.csv",
    #"urdfpath": "res/flyhopper_rwz/urdf/flyhopper_rwz.urdf",
    "linklengths": [.1, .27, .27, .1, .17, .0205],
    "k": 5000,  # wbc gain
    "k_g": 45,  # inv kin gain
    "k_gd": 45*0.02,
    "k_a": 1,
    "k_ad": 1*0.08,
    "springpolarity": 1,
    "hconst": 0.27
}

design = {
    "model": "design",
    "controllerclass": wbc_parallel,
    "legclass": leg_parallel,
    "csvpath": "res/flyhopper_robot/urdf/flyhopper_robot.csv",
    "urdfpath": "res/flyhopper_robot/urdf/flyhopper_robot.urdf",
    "linklengths": [.1, .3, .3, .1, .2, .0205],
    "k": 4000,
    "k_g": 37.5,
    "k_gd": 37.5*0.02,
    "k_a": 1,
    "k_ad": 1*0.08,
    "springpolarity": 1,
    "hconst": 0.3
}

parallel = {
    "model": "parallel",
    "controllerclass": wbc_parallel,
    "legclass": leg_parallel,
    "csvpath": "res/flyhopper_parallel/urdf/flyhopper_parallel.csv",
    "urdfpath": "res/flyhopper_parallel/urdf/flyhopper_parallel.urdf",
    "linklengths": [.15, .3, .3, .15, .15, 0],
    "k": 4000,
    "k_g": 70,
    "k_gd": 70*0.02,
    "k_a": 2,
    "k_ad": 2*0.08,
    "springpolarity": -1,
    "hconst": 0.3
}

serial = {
    "model": "serial",
    "controllerclass": wbc_serial,
    "legclass": leg_serial,
    "csvpath": "res/flyhopper_mockup/urdf/flyhopper_mockup.csv",
    "urdfpath": "res/flyhopper_mockup/urdf/flyhopper_mockup.urdf",
    "linklengths": [.3, .3],
    "k": 4000,
    "k_g": 70,
    "k_gd": 70*0.02,
    "k_a": 2,
    "k_ad": 2*0.08,
    "springpolarity": 0,
    "hconst": 0.3
}

belt = {
    "model": "belt",
    "controllerclass": wbc_serial,
    "legclass": leg_belt,
    "csvpath": "res/flyhopper_mockup/urdf/flyhopper_mockup.csv",
    "urdfpath": "res/flyhopper_mockup/urdf/flyhopper_mockup.urdf",
    "linklengths": [.3, .3],
    "k_g": 15,
    "k_gd": 15*0.02,
    "k_a": 0.5,
    "k_ad": 0.5*0.08,
    "springpolarity": 0,
    "hconst": 0.3
}