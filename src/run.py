"""
Copyright (C) 2021 Benjamin Bokser
"""
import argparse

import param
from robotrunner import Runner


dt = 1e-3

parser = argparse.ArgumentParser()

parser.add_argument("model", help="choose the robot model",
                    choices=['design_rw', 'design', 'serial', 'parallel', 'belt'], type=str)
parser.add_argument("ctrl", help="'wbc_raibert, wbc_vert, wbc_static, invkin_vert, or invkin_static",
                    choices=['wbc_raibert', 'wbc_vert', 'wbc_static', 'invkin_vert', 'invkin_static'],
                    type=str)
parser.add_argument("--plot", help="whether or not you would like to plot results", action="store_true")
parser.add_argument("--fixed", help="fixed base: True or False", action="store_true")
parser.add_argument("--spring", help="add spring: True or False", action="store_true")
parser.add_argument("--record", help="record: True or False", action="store_true")
parser.add_argument("--scale", help="change scale of robot (doesn't change mass)", type=float, default=1)
parser.add_argument("--recalc", help="re-calculate leg data",  action="store_true")
parser.add_argument("--gravoff", help="turn gravity off in sim", action="store_true")
parser.add_argument("--runtime", help="sim run time in ms (integer)", type=int, default=10000)
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

if args.recalc:
    recalc = True
else:
    recalc = False

if args.gravoff:
    gravoff = True
else:
    gravoff = False

print("\n")
print("model = ", args.model)
print("ctrl = ", args.ctrl)
print("\n")

if args.model == "design_rw":
    model = param.design_rw
elif args.model == "design":
    model = param.design
elif args.model == "parallel":
    model = param.parallel
elif args.model == "serial":
    model = param.serial
elif args.model == "belt":
    model = param.belt
else:
    raise NameError('INVALID MODEL')

runner = Runner(dt=dt, plot=plot, model=model, ctrl_type=args.ctrl, fixed=fixed, spring=spring, record=record,
                scale=args.scale, recalc=recalc, gravoff=gravoff, total_run=args.runtime, gain=model["k"])
runner.run()
