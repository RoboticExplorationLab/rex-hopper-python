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

```shell 
git clone https://github.com/LocknutBushing/spryped.git
```  

2. Make sure both Python 3.7 and pip are installed.

```shell
sudo apt install python3.7
sudo apt-get install python3-pip
python3.7 -m pip install --upgrade pip
```

2. I recommend setting up a virtual environment for this, as it requires the use of a number of specific Python packages.

```shell
sudo apt-get install python3.7-venv
cd flyhopper/src
python3.7 -m venv env
```
For more information on virtual environments: https://docs.python.org/3/library/venv.html
    
3. Activate the virtual environment, and then install numpy, scipy, matplotlib, sympy, transforms3d, pybullet, cvxpy, and argparse.

```shell
source env/bin/activate
python3.7 -m pip install numpy scipy matplotlib sympy transforms3d pybullet cvxpy argparse
```
Don't use sudo here if you can help it, because it may modify your path and install the packages outside of the venv.

Here is the [PyBullet tutorial](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3).

---

## Info

Here is some example code, which runs a simulation of a hopper standing with reaction wheels:

```shell
cd flyhopper/src
source env/bin/activate
python3.7 run.py design_rw static_invkin
```

### Argparse Arguments
There are two required argparse arguments: model and control. In addition, there are a number of optional arguments.

#### model
Warning: Some of the older models have deprecated functionality.
Choices:
* design_rw (The design-based model with reaction wheels, unconstrained with the world frame.)
* design (The design-based parallel leg.)
* serial (A simple serial double-pendulum leg.)
* parallel (A simple parallel leg.)
* belt (A simple double-pendulum leg with a belt constraint that allows it to be driven with one motor. Mostly deprecated.)

#### ctrl
Choices:
* simple_invkin (Jumps up and down using simple inverse kinematics and PID.)
* static_invkin (Stands still.)
* wbc_cycle (Uses whole-body control to jump up and down.)

#### optional arguments
* --plot (Use Matplotlib to plot parameters.)
* --fixed (Fixes the base of the robot to the world frame.)
* --spring (Adds a parallel spring to a parallel leg. Does not work with serial or belt models.)
* --record (Records video of sim.)
* --scale (Changes size of the robot. Default=1.)
* --gravoff (Turn off gravity.)

### Scaling Analysis
There's also an iterative analysis that checks how changing the size of the robot affects flight time. It also checks a range of PD control gains for each scale.

```shell
cd flyhopper/src
source env/bin/activate
python3.7 run_scaling.py
```
