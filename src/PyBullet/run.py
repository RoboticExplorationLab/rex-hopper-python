"""
Copyright (C) 2020 Benjamin Bokser
"""
import argparse

from robotrunner import Runner


dt = 1e-3

parser = argparse.ArgumentParser()

parser.add_argument("model", help="design, serial, parallel, or belt", type=str)
parser.add_argument("ctrl", help="simple_invkin, static_invkin or wbc_cycle", type=str)
parser.add_argument("--plot", help="whether or not you would like to plot results", action="store_true")
parser.add_argument("--fixed", help="fixed base: True or False", action="store_true")
parser.add_argument("--spring", help="add spring: True or False", action="store_true")
parser.add_argument("--record", help="add spring: True or False", action="store_true")
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

if args.model == 'parallel' or args.model == 'belt' or args.model == 'design':
    if args.ctrl == 'wbc_cycle':
        print("WARNING: This won't work")

print("\n")
print("model = ", args.model)
print("ctrl = ", args.ctrl)
print("\n")

runner = Runner(dt=dt, plot=plot, model=args.model, ctrl_type=args.ctrl, fixed=fixed, spring=spring, record=record)
runner.run()
