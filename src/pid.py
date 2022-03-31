"""
Copyright (C) 2022 Benjamin Bokser
"""
import numpy as np
import utils

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


class PIDn:

    def __init__(self, kp, ki, kd, dt=1e-3,  **kwargs):

        # n-Dimensional PID controller
        n = np.shape(kp)[0]  # get size
        self.kp = np.zeros((n, n))
        self.kd = np.zeros((n, n))
        self.ki = np.zeros((n, n))
        np.fill_diagonal(self.kp, kp)
        np.fill_diagonal(self.ki, ki)
        np.fill_diagonal(self.kd, kd)
    
        self.dt = dt
        self.inp_prev = np.zeros(n)  # previous measurement
        self.err_sum = np.zeros(n)

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

    def pid_control_wrap(self, inp, setp):
        # pid control for angles (wraps error to pi)
        dt = self.dt
        kp = self.kp
        ki = self.ki
        kd = self.kd
        err_sum = self.err_sum
        err = utils.wrap_to_pi(inp - setp)
        inp_diff = (inp - self.inp_prev) / dt
        u = kp @ err + ki @ err_sum * dt + kd @ inp_diff
        self.inp_prev = inp
        self.err_sum += err

        return u
