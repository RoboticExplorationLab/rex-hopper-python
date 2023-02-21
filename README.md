# REx Hopper

## Table of Contents

- [Intro](#intro)
- [Setup](#setup)
- [Info](#info)

---
## Intro

This repository mainly contains Python code for simulation of the REx Hopper in PyBullet. There are also supplementary Jupyter notebooks and matlab files in /scripts.

---

## Setup

1. Clone this directory wherever you want.

2. Make sure both Python 3.8 and pip are installed.

```shell
sudo apt install python3.8
sudo apt-get install python3-pip
python3.8 -m pip install --upgrade pip
```

2. I recommend setting up a virtual environment for this, as it requires the use of a number of specific Python packages.

```shell
sudo apt-get install python3.8-venv
cd rex_hopper_python/src
python3.8 -m venv env
```
For more information on virtual environments: https://docs.python.org/3/library/venv.html
    
3. Activate the virtual environment, and then install numpy, scipy, matplotlib, sympy, transforms3d, pybullet, cvxpy, argparse, and more.

```shell
source env/bin/activate
python3.8 -m pip install numpy scipy matplotlib sympy transforms3d pybullet cvxpy argparse dill SciencePlots
```
Don't use sudo here if you can help it, because it may modify your path and install the packages outside of the venv.

Here is the [PyBullet tutorial](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3).

---

## Info

Here is some example code, which runs a simulation of a hopper standing with reaction wheels:

```shell
cd rex_hopper_python/src
source env/bin/activate
python3.8 run.py design_rw ik_static --N_run=5000 --plot
```

### Argparse Arguments
There are two required argparse arguments: model and control. In addition, there are a number of optional arguments.

### model
Choices:
* `design_rw` (The design-based model with reaction wheels, unconstrained with the world frame.)
* `design_rw_alt`

### ctrl
Choices:
* `mpc` (model predictive controller)
* `wbc_raibert` (raibert hopping with whole body leg controller)
* `wbc_vert` (hopping straight up and down with whole body leg controller)
* `wbc_static` (standing still with whole body leg controller)
* `ik_vert` (Jumps up and down using simple inverse kinematics and PID.)
* `ik_static` (Stands using simple inverse kinematics and PID.)

### optional arguments
* `--plot` (Use Matplotlib to plot parameters.)
* `--fixed` (Fixes the base of the robot to the world frame.)
* `--spring` (Adds a parallel spring to a parallel leg. Does not work with serial or belt models.)
* `--record` (Records video of sim.)
* `--recalc` (Recalculate leg kinematics, dynamics, jacobian, etc. Saves new solution for future runs.)
* `--direct` (Turn off gravity.)
