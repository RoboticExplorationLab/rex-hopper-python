import numpy as np

def rw_control(x_ref, theta, omega):
    """
    simple reaction wheel control
    """
    # TODO: Add speed control inner PID loop
    kp = 0.1
    kd = np.copy(kp)*0.05
    tau_xdes = kp*(theta[0] - x_ref[0]) + kd*(omega[0] - x_ref[6])
    tau_ydes = kp*(theta[1] - x_ref[1]) + kd*(omega[1] - x_ref[7])
    u_rw = np.zeros(2)
    u_rw[0] = (tau_xdes + tau_ydes) / (2 * np.sin(45))
    u_rw[1] = (tau_xdes - tau_ydes) / (2 * np.sin(45))
    return u_rw