import numpy as np


class Actuator:
    def __init__(self, i_max, v_max, gr_out, tau_stall, omega_max, **kwargs):

        self.i_max = i_max
        self.v_max = v_max
        self.gr_out = gr_out
        self.tau_stall = tau_stall
        self.omega_max = omega_max  # 190 * 7 * (2 * np.pi / 60)  # rated speed, rpm to radians/s

        # self.kt_m = tau_stall / i_max
        self.kt_m = v_max/omega_max
        # r = (v_max ** 2) / (omega_max * tau_stall)
        self.r = self.kt_m * v_max / tau_stall

    def actuate(self, i, omega, printval=False):
        """
        Motor Dynamics
        i = current, Amps
        q_dot = angular velocity of link, after gear ratio (rad/s)
        gr = gear ratio of base actuator (which max torque rating is based on)
        gr_out = gear ratio of output after additional gearing is added
        """
        v_max = self.v_max
        gr_out = self.gr_out
        tau_stall = self.tau_stall
        kt_m = self.kt_m
        r = self.r
        omega_max = self.omega_max  # motor rated speed, rad/s
        # omega = abs(q_dot * gr_out)  # angular speed (NOT velocity) of motor in rad/s
        # kt_m1 = tau_stall * r / (v_max*gr)  # torque constant of the motor, Nm/amp. == v_max/omega_max
        tau_m = kt_m * i
        tau_max_m = abs(- abs(omega) * (kt_m ** 2) + v_max * kt_m) / r  # max motor torque for given speed
        if printval == True:
            print(omega_max, omega)
            # print(omega * (kt_m ** 2), v_max * kt_m)
        tau_max_m = np.clip(tau_max_m, -tau_stall, tau_stall)
        tau_m = np.clip(tau_m, -tau_max_m, tau_max_m)  # ensure motor torque remains within torque-speed curve

        return tau_m * gr_out  # actuator output torque

    def actuate_sat(self, i, omega):
        """
        Uses simple inverse torque-speed relationship
        """
        gr_out = self.gr_out
        omega_max = self.omega_max  # motor rated speed, rad/s
        tau_stall = self.tau_stall
        # omega = abs(q_dot * gr_out)  # angular speed (NOT velocity) of motor in rad/s
        tau_max = (1-abs(omega)/omega_max) * tau_stall * gr_out
        return np.clip(i, -tau_max, tau_max)
