import numpy as np


def hat(w):
    # skew-symmetric
    return np.array([[0, - w[2], w[1]],
                     [w[2], 0, - w[0]],
                     [-w[1], w[0], 0]])


def L(Q):
    LQ = np.zeros((4, 4))
    LQ[0, 0] = Q[0]
    LQ[0, 1:4] = - np.transpose(Q[1:4])
    LQ[1:4, 0] = Q[1:4]
    LQ[1:4, 1:4] = Q[0] * np.eye(3) + hat(Q[1:4])
    return LQ


def R(Q):
    RQ = np.zeros((4, 4))
    RQ[0, 0] = Q[0]
    RQ[0, 1:4] = - np.transpose(Q[1:4])
    RQ[1:4, 0] = Q[1:4]
    RQ[1:4, 1:4] = Q[0] * np.eye(3) - hat(Q[1:4])
    return RQ


H = np.zeros((4, 3))
H[1:4, 0:4] = np.eye(3)

T = np.zeros((4, 4))
np.fill_diagonal(T, [1.0, -1.0, -1.0, -1.0])


def Q_inv(Q):
    # Quaternion inverse
    Qinv = T @ Q
    return Qinv


def Expq(phi):
    # The quaternion exponential map ϕ → q
    Q = np.zeros(4)
    theta = np.linalg.norm(phi)
    Q[0] = np.cos(theta / 2)
    Q[1:4] = 0.5 * phi @ np.sinc(theta / (2 * np.pi))
    Q = Q / (np.linalg.norm(Q))  # re-normalize
    return Q


def Z(Q, p):
    # Rotate a position vector p by a quaternion Q
    return H.T @ R(Q).T @ L(Q) @ H @ p


def anglesolve(Q):
    # convert arbitrary quaternion to unsigned angle
    return 2 * np.arctan2(np.linalg.norm(Q[1:4]), Q[0])


def angle_y(Q1, Q2):
    # signed angle about y axis of y-axis-constrained quaternions
    Q12 = L(Q1).T @ Q2
    Q12 = Q12 / (np.linalg.norm(Q12))
    return 2 * np.arcsin(Q12[2])


def angle_z(Q1, Q2):
    # signed angle about z axis of z-axis-constrained quaternions
    Q12 = L(Q1).T @ Q2
    Q12 = Q12 / (np.linalg.norm(Q12))
    return 2 * np.arcsin(Q12[3])


def z_rotate(Q_in, z):
    # rotate quaternion about its z-axis by specified angle "z"
    # and get rotation about x-axis of that (confusing, I know)
    Q_z = np.array([np.cos(z / 2), 0, 0, np.sin(z / 2)]).T
    Q_res = L(Q_z).T @ Q_in
    Q_res = Q_res / (np.linalg.norm(Q_res))
    theta_res = 2 * np.arcsin(Q_res[1])  # x-axis of rotated body quaternion
    return theta_res


def vec_to_quat(v2):
    # conversion of line vector to quaternion rotation b/t it and a datum vector v1
    v1 = np.array([1, 0, 0])  # datum vector, chosen as aligned with x-axis (front facing)
    Q = np.zeros(4)
    Q[0] = np.sqrt((np.linalg.norm(v1)**2)*(np.linalg.norm(v2)**2)) + np.dot(v1, v2)
    Q[1:4] = np.cross(v1, v2)
    Q = Q / np.linalg.norm(Q)
    return Q


def vec_to_quat2(v2):
    # conversion of line vector to quaternion rotation b/t it and a datum vector v1
    # alternative version
    v1 = np.array([0, 0, -1])  # datum vector, chosen as aligned with z-axis (representing leg direction)
    u1 = v1 / np.linalg.norm(v1)
    Q = np.zeros(4)
    if np.linalg.norm(v2) == 0 or np.isnan(v2).any():  # if input vector is zero or NaN, make Q default
        Q[0] = 1
    else:
        u2 = v2 / np.linalg.norm(v2)
        if np.array_equal(u1, -u2):
            Q[1:4] = np.linalg.norm(np.cross(v1, v2))
        else:
            u_half = (u1 + u2)/np.linalg.norm(u1 + u2)
            Q[0] = np.dot(u1, u_half)
            Q[1:4] = np.cross(u1, u_half)
            Q = Q / np.linalg.norm(Q)
    return Q_inv(Q)


# --- from Shuo's quadruped code --- #
def quat2rot(Q):
    w, x, y, z = Q
    R = np.array([[2 * (w ** 2 + x ** 2) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 2 * (w ** 2 + y ** 2) - 1, 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w ** 2 + z ** 2) - 1]])
    return R


def euler2rot(euler):
    x, y, z = euler
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(x), -np.sin(x)],
                   [0.0, np.sin(x), np.cos(x)]])
    Ry = np.array([[np.cos(y), 0.0, np.sin(y)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(y), 0.0, np.cos(y)]])
    Rz = np.array([[np.cos(z), -np.sin(z), 0.0],
                   [np.sin(z), np.cos(z), 0.0],
                   [0.0, 0.0, 1.0]])
    R = Rz @ Ry @ Rx
    return R


def quat2euler(quat):
    w, x, y, z = quat
    y_sqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y_sqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y_sqr + z * z)
    Z = np.arctan2(t3, t4)

    result = np.zeros(3)
    result[0] = X
    result[1] = Y
    result[2] = Z

    return result


def euler2quat(euler):
    x, y, z = euler
    z = z / 2.0
    y = y / 2.0
    x = x / 2.0
    cz = np.cos(z)
    sz = np.sin(z)
    cy = np.cos(y)
    sy = np.sin(y)
    cx = np.cos(x)
    sx = np.sin(x)
    result = np.array([
        cx * cy * cz - sx * sy * sz,
        cx * sy * sz + cy * cz * sx,
        cx * cz * sy - sx * cy * sz,
        cx * cy * sz + sx * cz * sy])
    if result[0] < 0:
        result = -result
    return result


def euler2quat_order(euler, order):
    r, p, y = euler
    y = y / 2.0
    p = p / 2.0
    r = r / 2.0
    c3 = np.cos(y)
    s3 = np.sin(y)
    c2 = np.cos(p)
    s2 = np.sin(p)
    c1 = np.cos(r)
    s1 = np.sin(r)
    if order == 'XYZ':
        result = np.array([
            c1 * c2 * c3 - s1 * s2 * s3,
            s1 * c2 * c3 + c1 * s2 * s3,
            c1 * s2 * c3 - s1 * c2 * s3,
            c1 * c2 * s3 + s1 * s2 * c3])
        return result
    elif order == 'YXZ':
        result = np.array([
            c1 * c2 * c3 + s1 * s2 * s3,
            s1 * c2 * c3 + c1 * s2 * s3,
            c1 * s2 * c3 - s1 * c2 * s3,
            c1 * c2 * s3 - s1 * s2 * c3])
        return result
    elif order == 'ZXY':
        result = np.array([
            c1 * c2 * c3 - s1 * s2 * s3,
            s1 * c2 * c3 - c1 * s2 * s3,
            c1 * s2 * c3 + s1 * c2 * s3,
            c1 * c2 * s3 + s1 * s2 * c3])
        return result
    elif order == 'ZYX':
        result = np.array([
            c1 * c2 * c3 + s1 * s2 * s3,
            s1 * c2 * c3 - c1 * s2 * s3,
            c1 * s2 * c3 + s1 * c2 * s3,
            c1 * c2 * s3 - s1 * s2 * c3])
        return result
    elif order == 'YZX':
        result = np.array([
            c1 * c2 * c3 - s1 * s2 * s3,
            s1 * c2 * c3 + c1 * s2 * s3,
            c1 * s2 * c3 + s1 * c2 * s3,
            c1 * c2 * s3 - s1 * s2 * c3])
        return result
    elif order == 'XZY':
        result = np.array([
            c1 * c2 * c3 + s1 * s2 * s3,
            s1 * c2 * c3 - c1 * s2 * s3,
            c1 * s2 * c3 - s1 * c2 * s3,
            c1 * c2 * s3 + s1 * s2 * c3])
        return result