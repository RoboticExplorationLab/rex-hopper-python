"""
Copyright (C) 2021 Benjamin Bokser
"""

import numpy as np

def rw_control(dt, x_ref, theta, omega, err_sum, err_prev):
    """
    simple reaction wheel control
    """
    # TODO: Add speed control inner PID loop
    # print(x_ref[0:2], theta[0:2])
    ku = 25
    # Tu =
    kp = ku # 0.6*ku
    kd = 0 # kp*0.075
    ki = 0
    kpz = 50
    kdz = kpz*0.2

    err = theta[0:3] - x_ref[0:3]
    '''
    tau_xdes = kp * err[0] + kd * (omega[0] - x_ref[6]) + ki*err_sum[0]*ts
    tau_ydes = kp * err[1] + kd * (omega[1] - x_ref[7]) + ki*err_sum[1]*ts
    tau_zdes = kpz * err[2] + kdz * (omega[2] - x_ref[8]) + ki*err_sum[2]*ts
    '''
    err_diff = (err-err_prev)/dt
    tau_xdes = kp * err[0] + kd * err_diff[0] + ki*err_sum[0]*dt
    tau_ydes = kp * err[1] + kd * err_diff[1] + ki*err_sum[1]*dt
    tau_zdes = kpz * err[2] + kdz * err_diff[2] + ki*err_sum[2]*dt

    u_rw = np.zeros(3)
    u_rw[0] = (tau_xdes + tau_ydes) / (2 * np.sin(45))
    u_rw[1] = (tau_xdes - tau_ydes) / (2 * np.sin(45))
    u_rw[2] = tau_zdes

    err_sum += err
    err_prev = err
    return u_rw, err_sum, err_prev