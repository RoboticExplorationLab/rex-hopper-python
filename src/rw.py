"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np

import utils

def z_rotate(Q_in, z):
    # rotate quaternion about its z-axis by specified angle "z"
    # and get rotation about x-axis of that (confusing, I know)
    Q_z = np.array([np.cos(z / 2), 0, 0, np.sin(z / 2)]).T
    Q_res = utils.L(Q_z).T @ Q_in
    Q_res = Q_res / (np.linalg.norm(Q_res))
    theta_res = 2 * np.arcsin(Q_res[1])  # x-axis of rotated body quaternion
    return theta_res

def rw_control(dt, Q_ref, Q_base, err_sum, err_prev):
    """
    simple reaction wheel control
    TODO: Add actuator model
    TODO: Add speed control inner PID loop
    """
    a = -45 * np.pi / 180
    b = 45 * np.pi / 180

    ref_1 = z_rotate(Q_ref, a)
    ref_2 = z_rotate(Q_ref, b)
    # setp = np.array([-3.35*np.pi/180, 3.35*np.pi/180, 0])
    setp = np.array([ref_1 - 3.35 * np.pi / 180, ref_2 + 3.35 * np.pi / 180, 0])

    theta_1 = z_rotate(Q_base, a)
    theta_2 = z_rotate(Q_base, b)

    theta_3 = 2 * np.arcsin(Q_base[3])  # z-axis of body quaternion

    # print("angle = ", utils.anglesolve(utils.L(Q_1).T @ Q_2)*180/np.pi)

    theta = np.array([theta_1, theta_2, theta_3])

    ku = 180
    kp = np.zeros((3, 3))
    kd = np.zeros((3, 3))
    ki = np.zeros((3, 3))
    np.fill_diagonal(kp, [ku, -ku, 8])  # ku*0.6
    np.fill_diagonal(ki, [ku * 0, -ku * 0, 0])  # ku*3*tu/40
    np.fill_diagonal(kd, [ku * 0.0001, -ku * 0.0001, 8 * 0.002])  # ku*1.2/tu

    err = theta[0:3] - setp[0:3]

    err_diff = (err-err_prev)/dt
    u_rw = kp @ err + ki @ err_sum * dt + kd @ err_diff / dt

    err_sum += err
    err_prev = err
    return u_rw, err_sum, err_prev, theta, setp