"""
Copyright (C) 2020 Benjamin Bokser
"""
import argparse

from robotrunner import Runner


dt = 1e-3

parser = argparse.ArgumentParser()
parser.add_argument("plot", help="Whether or not you would like to plot results", type=str)
parser.add_argument("model", help="serial, parallel, or belt", type=str)
parser.add_argument("ctrl", help="simple_invkin, static_invkin or wbc_cycle", type=str)
args = parser.parse_args()

if args.plot == 'True':
    plot = True
else:
    plot = False

if args.model == 'parallel' or args.model == 'belt':
    if args.ctrl == 'wbc_cycle':
        print("WARNING: This won't work")

print("\n")
print("plot = ", plot)
print("model = ", args.model)
print("ctrl = ", args.ctrl)
print("\n")

runner = Runner(dt=dt, plot=plot, model=args.model, ctrl_type=args.ctrl)
runner.run()
