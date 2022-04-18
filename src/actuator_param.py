"""
Copyright (C) 2022 Benjamin Bokser
"""

import numpy as np


actuator_mn3110kv700 = {
    "name": "mn3110",
    "v_max": 24,
    "kt": 1/(700 * (2 * np.pi / 60)),
    "omega_max": 8600 * (2 * np.pi / 60),
    "tau_max": None,
    "r": 0.092,
    "i_max": 26,
    "gr": 1
}


actuator_u8 = {
    "name": "u8",
    "v_max": 48,
    "kt": 8.4/100,
    "omega_max": 3700 * (2 * np.pi / 60),
    "tau_max": 2.8,
    "r": 0.186,
    "i_max": 31,
    "gr": 1
}

actuator_mn1005kv90 = {
    "name": "mn1005",
    "v_max": 48,
    "kt": 8.4/90,
    "omega_max": 3200 * (2 * np.pi / 60),
    "tau_max": 3.4,
    "r": 0.168,
    "i_max": 40,
    "gr": 1
}

actuator_r80kv110 = {
    "name": "r80",
    "v_max": 48,
    "kt": 0.0868,
    "omega_max": 4600 * (2 * np.pi / 60),
    "tau_max": 4,
    "r": 0.125,
    "i_max": 46,
    "gr": 1
}

actuator_8318 = {
    "name": "8318",
    "v_max": 48,
    "kt": 8.4/100,
    "omega_max": 3840 * (2 * np.pi / 60),
    "tau_max": 4.71,
    "r": 0.055,
    "i_max": 60,
    "gr": 1
}

actuator_rmdx10 = {
    "name": "RMD-X10",
    "v_max": 48,
    "kt": 1.73/7,
    "omega_max": 250 * 7 * (2 * np.pi / 60),
    "tau_max": 50/7,
    "r": 0.3,
    "i_max": 30,
    "gr": 7
}

actuator_ea110 = {
    "name": "EA110-100KV",
    "v_max": 48,
    "kt": 8.4/100,
    "omega_max": 3490 * (2 * np.pi / 60),
    "tau_max": 11.24,
    "r": 33/1000,
    "i_max": 120,
    "gr": 1
}

actuator_r100kv90 = {
    "name": "R100-90KV",
    "v_max": 48,
    "kt": 0.106,
    "omega_max": 3800 * (2 * np.pi / 60),
    "tau_max": 11.24,
    "r": 51/1000,
    "i_max": 104,
    "gr": 1
}
