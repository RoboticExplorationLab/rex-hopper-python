"""
Copyright (C) 2020 Benjamin Bokser

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import argparse

from robotrunner import Runner


dt = 1e-3

parser = argparse.ArgumentParser()
parser.add_argument("plot", help="Whether or not you would like to plot results", type=str)
parser.add_argument("model", help="serial, parallel, or belt", type=str)
args = parser.parse_args()

if args.plot == 'True':
    plot = True
else:
    plot = False

if args.model == 'serial':
    ctrl = 'wbc_cycle'
else:
    ctrl = 'simple_invkin'  # necessary because other models do not have dynamics implemented yet

print("\n")
print("plot = ", plot)
print("model = ", args.model)
print("\n")

runner = Runner(dt=dt, plot=plot, model=args.model, ctrl_type=ctrl)
runner.run()
