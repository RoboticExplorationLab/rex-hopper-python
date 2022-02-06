"""
Copyright (C) 2022 Benjamin Bokser
"""

import numpy as np


actuator_u8 = {
    "name": "8318",
    "kt": 8.4/100,
    "omega_max": 3700 * (2 * np.pi / 60),
    "tau_stall": 2.8,
    "r": 0.186,
    "i_max": 31,
    "i_stall": 31,
    "gr": 1
}

actuator_8318 = {
    "name": "8318",
    "kt": 8.4/100,
    "omega_max": 3840 * (2 * np.pi / 60),
    "tau_stall": 4.71,
    "r": 0.055,
    "i_max": 60,
    "i_stall": 57,
    "gr": 1
}

actuator_rmdx10 = {
    "name": "RMD-X10",
    "kt": 1.73/7,
    "omega_max": 250 * 7 * (2 * np.pi / 60),
    "tau_stall": 50/7,
    "r": 0.3,
    "i_max": 20,
    "i_stall": 13,
    "gr": 7
}

actuator_ea110 = {
    "name": "EA110-100KV",
    "kt": 8.4/100,
    "omega_max": 3490 * (2 * np.pi / 60),
    "tau_stall": 11.24,
    "r": 33/1000,
    "i_max": 120,
    "i_stall": 120,
    "gr": 1
}

actuator_r10090kv = {
    "name": "R100-90KV",
    "kt": 0.106,
    "omega_max": 3800 * (2 * np.pi / 60),
    "tau_stall": 11.24,
    "r": 51/1000,
    "i_max": 104,
    "i_stall": 104,
    "gr": 1
}
