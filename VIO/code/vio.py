# %% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


# %% Functions

def skew(v):
    return np.array([[0,  -v[2],  v[1]],
                     [v[2],    0, -v[0]],
                     [-v[1], v[0],   0]])


def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    R = q.as_matrix()
    new_p = p + v * dt + (1 / 2) * (R @ (a_m - a_b) + g) * dt ** 2
    new_v = v + (R @ (a_m - a_b) + g) * dt
    new_q = q * Rotation.from_rotvec(((w_m - w_b) * dt).flatten())

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    I = np.identity(3)
    R = q.as_matrix()

    Fx = np.identity(18)
    Fx[0:3, 3:6] = I * dt
    Fx[3:6, 6:9] = -R @ skew((a_m-a_b).flatten())*dt
    Fx[3:6, 9:12] = -R * dt
    Fx[3:6, 15:18] = I * dt
    Fx[6:9, 6:9] = Rotation.from_rotvec(
        ((w_m - w_b) * dt).flatten()).as_matrix().T
    Fx[6:9, 12:15] = -I * dt

    Qi = np.identity(12)
    Qi[0:3, 0:3] = I * (accelerometer_noise_density ** 2) * (dt ** 2)
    Qi[3:6, 3:6] = I * (gyroscope_noise_density ** 2) * (dt ** 2)
    Qi[6:9, 6:9] = I * (accelerometer_random_walk ** 2) * dt
    Qi[9:12, 9:12] = I * (gyroscope_random_walk ** 2) * dt

    Fi = np.zeros((18, 12))
    Fi[3:15, :] = np.identity(12)

    # return an 18x18 covariance matrix
    return Fx @ error_state_covariance @ Fx.T + Fi @ Qi @ Fi.T


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    R = q.as_matrix()
    Pc = R.T @ (Pw - p)
    innovation = uv - Pc[0:2] / Pc[2]

    # extract
    Xc = Pc[0, 0]
    Yc = Pc[1, 0]
    Zc = Pc[2, 0]
    dzdP = np.array([[1/Zc, 0, - Xc / (Zc ** 2)],
                     [0, 1/Zc, - Yc / (Zc ** 2)]])
    dPdtheta = skew(Pc.flatten())
    dPdp = - R.T
    # fill
    Ht = np.zeros((2, 18))
    Ht[:, :3] = dzdP @ dPdp
    Ht[:, 6:9] = dzdP @ dPdtheta

    inv = np.linalg.inv(Ht @ error_state_covariance @ Ht.T + Q)
    Kt = error_state_covariance @ Ht.T @ inv  # 18x2 matrix
    tmp = np.identity(18) - Kt @ Ht  # 18x18 matrix
    error_state_covariance = tmp @ error_state_covariance @ tmp.T
    error_state_covariance += Kt @ Q @ Kt.T

    if np.linalg.norm(innovation) < error_threshold:

        dx = Kt @ innovation  # 18x1 vector

        p += dx[:3]
        v += dx[3:6]
        q = q * Rotation.from_rotvec(dx[6:9].flatten())
        a_b += dx[9:12]
        w_b += dx[12:15]
        g += dx[15:18]

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
