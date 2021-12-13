"""
Copyright (C) 2021 Benjamin Bokser
"""
import argparse

import leg_serial
import leg_parallel
import leg_belt
import wbc_parallel
import wbc_serial
from robotrunner import Runner


dt = 1e-3

parser = argparse.ArgumentParser()

parser.add_argument("model", help="choose the robot model",
                    choices=['design_rw', 'design', 'serial', 'parallel', 'belt'], type=str)
parser.add_argument("ctrl", help="simple_invkin, static_invkin or wbc_cycle",
                    choices=['wbc_cycle', 'simple_invkin', 'static_invkin'], type=str)
parser.add_argument("--plot", help="whether or not you would like to plot results", action="store_true")
parser.add_argument("--fixed", help="fixed base: True or False", action="store_true")
parser.add_argument("--spring", help="add spring: True or False", action="store_true")
parser.add_argument("--record", help="record: True or False", action="store_true")
parser.add_argument("--scale", help="change scale of robot (doesn't change mass)", type=float, default=1)
parser.add_argument("--gravoff", help="turn gravity off in sim", action="store_true")
args = parser.parse_args()

if args.plot:
    plot = True
else:
    plot = False

if args.fixed:
    fixed = True
else:
    fixed = False

if args.spring:
    spring = True
else:
    spring = False

if args.record:
    record = True
else:
    record = False

if args.gravoff:
    gravoff = True
else:
    gravoff = False

print("\n")
print("model = ", args.model)
print("ctrl = ", args.ctrl)

print("\n")

design_rw = {
    "model": "design_rw",
    "controllerclass": wbc_parallel,
    "legclass": leg_parallel,
    "csvpath": "res/flyhopper_rw/urdf/flyhopper_rw.csv",
    "urdfpath": "res/flyhopper_rw/urdf/flyhopper_rw.urdf",
    "linklengths": [.1, .3, .3, .1, .2, .0205],
    "k_kin": 37.5,
    "springpolarity": 1
}

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

parallel = {
    "model": "parallel",
    "controllerclass": wbc_parallel,
    "legclass": leg_parallel,
    "csvpath": "res/flyhopper_parallel/urdf/flyhopper_parallel.csv",
    "urdfpath": "res/flyhopper_parallel/urdf/flyhopper_parallel.urdf",
    "linklengths": [.15, .3, .3, .15, .15, 0],
    "k_kin": 70,
    "springpolarity": -1
}

serial = {
    "model": "serial",
    "controllerclass": wbc_serial,
    "legclass": leg_serial,
    "csvpath": "res/flyhopper_mockup/urdf/flyhopper_mockup.csv",
    "urdfpath": "res/flyhopper_mockup/urdf/flyhopper_mockup.urdf",
    "linklengths": [.3, .3],
    "k_kin": 70,
    "springpolarity": 0
}

belt = {
    "model": "belt",
    "controllerclass": wbc_serial,
    "legclass": leg_belt,
    "csvpath": "res/flyhopper_mockup/urdf/flyhopper_mockup.csv",
    "urdfpath": "res/flyhopper_mockup/urdf/flyhopper_mockup.urdf",
    "linklengths": [.3, .3],
    "k_kin": 15,
    "springpolarity": 0
}

if args.model == "design_rw":
    model = design_rw
    import leg_parallel
    import wbc_parallel
elif args.model == "design":
    model = design
    import leg_parallel
    import wbc_parallel
elif args.model == "parallel":
    model = parallel
    import leg_parallel
    import wbc_parallel
elif args.model == "serial":
    model = serial
    import leg_serial
    import wbc_serial
elif args.model == "belt":
    model = belt
    import leg_belt
else:
    raise NameError('INVALID MODEL')

runner = Runner(dt=dt, plot=plot, model=model, ctrl_type=args.ctrl, fixed=fixed,
                spring=spring, record=record, scale=args.scale, gravoff=gravoff)
runner.run()
