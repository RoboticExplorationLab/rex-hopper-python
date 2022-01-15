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


def rw_control(pid_torque, pid_vel, Q_ref, Q_base, qrw_dot):
    """
    simple reaction wheel control w/ derivative on measurement
    """
    a = -45 * np.pi / 180
    b = 45 * np.pi / 180

    ref_1 = z_rotate(Q_ref, a)
    ref_2 = z_rotate(Q_ref, b)
    # setp = np.array([ref_1 - 3.35 * np.pi / 180, ref_2 + 3.35 * np.pi / 180, 0])
    setp = np.array([ref_1 - 4 * np.pi / 180, ref_2 + 4 * np.pi / 180, 0])

    theta_1 = z_rotate(Q_base, a)
    theta_2 = z_rotate(Q_base, b)

    theta_3 = 2 * np.arcsin(Q_base[3])  # z-axis of body quaternion

    theta = np.array([theta_1, theta_2, theta_3])

    u_vel = pid_vel.pid_control(inp=qrw_dot.flatten(), setp=np.zeros(3))
    # u_tau = pid_torque.pid_control(inp=theta + u_vel, setp=setp)  # Cascaded PID Loop
    # return u_tau, theta + u_vel, setp
    u_tau = pid_torque.pid_control(inp=theta, setp=setp - u_vel)  # Cascaded PID Loop

    return u_tau, theta, setp - u_vel