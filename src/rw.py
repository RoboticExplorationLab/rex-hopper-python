"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np

import utils

def rw_control(dt, Q_base, err_sum, err_prev):
    """
    simple reaction wheel control
    TODO: Add speed control inner PID loop
    """
    x_ref = np.zeros(3)

    a = -45 * np.pi / 180
    b = 45 * np.pi / 180
    Q_a = np.array([np.cos(a / 2), 0, 0, np.sin(a / 2)]).T
    Q_1 = utils.L(Q_a) @ Q_base
    Q_1 = Q_1 / (np.linalg.norm(Q_1))
    theta_1 = 2 * np.arcsin(Q_1[1]) # x-axis of rotated body quaternion

    Q_b = np.array([np.cos(b / 2), 0, 0, np.sin(b / 2)]).T
    Q_2 = utils.L(Q_b) @ Q_base
    Q_2 = Q_2 / (np.linalg.norm(Q_2))
    theta_2 = 2 * np.arcsin(Q_2[1])  # x-axis of rotated body quaternion

    theta_3 = 2 * np.arcsin(Q_base[3])  # z-axis of body quaternion

    theta = np.array([theta_1, theta_2, theta_3])

    ku = 20

    kp = np.zeros((3, 3))
    kd = np.zeros((3, 3))
    ki = np.zeros((3, 3))

    np.fill_diagonal(kp, [-ku, ku, 10])  # ku*0.6
    np.fill_diagonal(kd, [0, 0, 10*0.002]) # ku*1.2/tu
    np.fill_diagonal(ki, [0, 0, 0])  # ku*3*tu/40

    err = theta[0:3] - x_ref[0:3]

    err_diff = (err-err_prev)/dt
    u_rw = kp @ err + kd @ err_diff / dt + ki @ err_sum * dt

    err_sum += err
    err_prev = err
    return u_rw, err_sum, err_prev, theta