"""
Copyright (C) 2022 Benjamin Bokser
"""
import numpy as np


class PID1:

    def __init__(self, kp, ki, kd, dt=1e-3, **kwargs):
        # 1-Dimensional PID controller
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.inp_prev = 0  # previous measurement
        self.err_sum = 0

    def update_k(self, kp, ki, kd):
        # Use this to update PID gains in real time
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def pid_control(self, inp, setp):
        dt = self.dt
        kp = self.kp
        ki = self.ki
        kd = self.kd
        err_sum = self.err_sum
        err = inp - setp
        inp_diff = (inp - self.inp_prev) / dt
        u = kp * err + ki * err_sum * dt + kd * inp_diff
        self.inp_prev = inp
        self.err_sum += err

        return u


class PID3:

    def __init__(self, kp, ki, kd, dt=1e-3,  **kwargs):

        # 3-Dimensional PID controller
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.dt = dt
        self.inp_prev = np.zeros(3)  # previous measurement
        self.err_sum = np.zeros(3)

    def update_k(self, kp, ki, kd):
        # Use this to update PID gains in real time
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def pid_control(self, inp, setp):
        dt = self.dt
        kp = self.kp
        ki = self.ki
        kd = self.kd
        err_sum = self.err_sum
        err = inp - setp
        inp_diff = (inp - self.inp_prev) / dt
        u = kp @ err + ki @ err_sum * dt + kd @ inp_diff
        self.inp_prev = inp
        self.err_sum += err

        return u
