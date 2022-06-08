"""
Copyright (C) 2021 Benjamin Bokser
"""
import argparse

import param
from robotrunner import Runner


dt = 1e-3

parser = argparse.ArgumentParser()

parser.add_argument("model", help="choose the robot model",
                    choices=['design_rw', 'design_rw_alt'], type=str)
parser.add_argument("ctrl", help="mpc, wbc_raibert, wbc_vert, wbc_static, ik_vert, or ik_static",
                    choices=['mpc', 'wbc_raibert', 'wbc_vert', 'wbc_static', 'ik_vert', 'ik_static'],
                    type=str)
parser.add_argument("--scale", help="change scale of robot (doesn't change mass)", type=float, default=1)
parser.add_argument("--N_run", help="sim run timesteps (integer)", type=int, default=10000)

parser.add_argument("--plot", help="Plot results", action="store_true")
parser.add_argument("--fixed", help="fixed base: True or False", action="store_true")
parser.add_argument("--spring", help="add spring: True or False", action="store_true")
parser.add_argument("--record", help="record: True or False", action="store_true")
parser.add_argument("--recalc", help="re-calculate leg data",  action="store_true")
parser.add_argument("--gravoff", help="turn gravity off in sim", action="store_true")
parser.add_argument("--direct", help="Run sim without graphics vis", action="store_true")

args = parser.parse_args()

plot = True if args.plot else False
fixed = True if args.fixed else False
spring = True if args.spring else False
record = True if args.record else False
recalc = True if args.recalc else False
gravoff = True if args.gravoff else False
direct = True if args.direct else False

print("\n")
print("model = ", args.model)
print("ctrl = ", args.ctrl)
print("\n")

if args.model == "design_rw":
    model = param.design_rw
elif args.model == "design_rw_alt":
    model = param.design_rw_alt
else:
    raise NameError('INVALID MODEL')

runner = Runner(dt=dt, plot=plot, model=model, ctrl_type=args.ctrl, fixed=fixed, spr=spring, record=record,
                scale=args.scale, recalc=recalc, gravoff=gravoff, direct=direct, N_run=args.N_run, gain=model["k"])
runner.run()
