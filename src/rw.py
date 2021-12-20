"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np

def rw_control(x_ref, theta, omega):
    """
    simple reaction wheel control
    """
    # TODO: Add speed control inner PID loop
    print(x_ref[0:2], theta[0:2])
    kp = 50
    kd = np.copy(kp)*0.15

    kpz = 50
    kdz = np.copy(kpz)*0.15
    # err = theta[0:3] - x_ref[0:3]
    tau_xdes = kp * (theta[0] - x_ref[0]) + kd * (omega[0] - x_ref[6])
    tau_ydes = kp * (theta[1] - x_ref[1]) + kd * (omega[1] - x_ref[7])
    tau_zdes = kpz * (theta[2] - x_ref[2]) + kdz * (omega[2] - x_ref[8])
    u_rw = np.zeros(3)
    u_rw[0] = (tau_xdes + tau_ydes) / (2 * np.sin(45))
    u_rw[1] = (tau_xdes - tau_ydes) / (2 * np.sin(45))
    u_rw[2] = tau_zdes
    return u_rw