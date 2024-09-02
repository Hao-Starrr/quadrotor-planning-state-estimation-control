import numpy as np
from scipy import io
from quaternion import Quaternion
import math
import matplotlib.pyplot as plt

# data files are numbered on the server.
# for exmaple imuRaw1.mat, imuRaw2.mat and so on.
# write a function that takes in an input number (1 through 6)
# reads in the corresponding imu Data, and estimates
# roll pitch and yaw using an unscented kalman filter


def estimate_rot(data_num=1):
    # load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    # vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3, :]
    gyro = imu['vals'][3:6, :]
    T = np.shape(imu['ts'])[1]

    # your code goes here
    print("Data loaded")
    print("Accelerometer data shape:", np.shape(accel))
    print("Gyroscope data shape:", np.shape(gyro))
    # print("Vicon data shape:", np.shape(vicon["rots"]))
    print("Number of samples:", T)

    # use dataset 1 to calculate the alpha and beta for all three axes

    # acceleration
    accel_alpha = [34.72781212749805, 34.45286490005449, 34.61822216398437]
    accel_beta = [511.3059001928169, 500.48395634970245, 500.55158211867194]

    # gyroscope
    gyro_alpha = [161.3406929934263, 127.5871499286338, 256.6501895128293]
    gyro_beta = [368.50612468209795, 382.1552029138575, 371.67932427361177]

    # convert data to physical units
    Ax, Ay, Az, Wx, Wy, Wz = convert2physical(accel, gyro, accel_alpha,
                                              accel_beta, gyro_alpha, gyro_beta)

    dt_list = imu['ts'][0, 1:] - imu['ts'][0, 0:-1]
    dt_list = np.hstack((0, dt_list))
    print("dt shape:", np.shape(dt_list))  # dt length T

    # the state is x = [q, w], 1x7
    # initialize state and covariance matrices at time 0
    # it define a distribution
    x_k_minus_1 = np.zeros(7)
    x_k_minus_1[0] = 1  # set initial q = [1 0 0 0]
    P_k_minus_1 = np.eye(6)*1.0
    print("shape of x_k_minus_1:",   np.shape(x_k_minus_1))

    # noise matrices, both 6x6 dimensions
    Q = np.eye(6)*0.2     # process noise
    R = np.eye(6)*0.1       # measurement noise

    roll, pitch, yaw = np.zeros(T), np.zeros(T), np.zeros(T)

    for i in range(int(T)):
        # main loop

        # generate sigma points
        # print(x_k_minus_1)
        # print(P_k_minus_1)
        # print(Q)
        Xi = generate_sigma_points(x_k_minus_1, P_k_minus_1, Q)
        # print(Xi)
        # predict
        Yi = process_model(Xi, dt_list[i])

        # calculate the means and covariance
        x_k_bar = calculate_mean(Yi)
        P_k_bar, W = calculate_covariance(x_k_bar, Yi)
        # print("x_k_bar:", x_k_bar)

        # generate sigma points again
        zero_covirance = np.zeros((6, 6))
        # Yi_sampled = generate_sigma_points(x_k_bar, P_k_bar, zero_covirance)
        Zi = measurement_model(Yi)
        # print(Zi)

        # equation 68, find the mean and covariance of Zi
        Z_mean = np.mean(Zi, axis=1)
        matrix = Zi - Z_mean.reshape(6, 1)
        Pzz = matrix @ matrix.T / 12
        # equation 69
        Pvv = Pzz + R
        # equation 71
        Pxz = (W @ matrix.T)/12.0
        # equation 72
        K = Pxz @ np.linalg.inv(Pvv)
        # equation 75
        P_k = P_k_bar - K @ Pvv @ K.T

        measurement = np.array([Ax[i], Ay[i], Az[i], Wx[i], Wy[i], Wz[i]])
        v_k = K @ (measurement - Z_mean)
        x_k = update_x_k(x_k_bar, v_k)

        x_k_q = Quaternion(x_k[0], x_k[1:4])
        euler_angles = x_k_q.euler_angles()
        roll[i] = euler_angles[0]
        pitch[i] = euler_angles[1]
        yaw[i] = euler_angles[2]

        x_k_minus_1 = x_k
        P_k_minus_1 = P_k
        print(x_k_minus_1)

    # roll, pitch, yaw are numpy arrays of length T
    return roll, pitch, yaw


def update_x_k(x_k_bar, v_k):

    # omega just plus
    x_k_bar[4:7] += v_k[3:6]

    # q need calculate
    v_k_q = Quaternion()
    v_k_q.from_axis_angle(v_k[0:3])

    x_k_bar_q = Quaternion(x_k_bar[0], x_k_bar[1:4])
    x_k_bar_q = v_k_q * x_k_bar_q
    x_k_bar_q.normalize()

    x_k_bar[0:4] = x_k_bar_q.q

    return x_k_bar


def digital2analog(raw_data, alpha, beta):
    value = (raw_data - beta) * 3300 / (1023 * alpha)
    return value


def convert2physical(accel, gyro, accel_alpha, accel_beta, gyro_alpha, gyro_beta):
    # change the axis
    Ax = -digital2analog(accel[0, :], accel_alpha[0], accel_beta[0])
    Ay = -digital2analog(accel[1, :], accel_alpha[1], accel_beta[1])
    Az = digital2analog(accel[2, :], accel_alpha[2], accel_beta[2])

    gyro_alpha_mean = np.mean(gyro_alpha)
    # Wz is at 0
    Wx = digital2analog(gyro[1, :], gyro_alpha_mean, gyro_beta[0])
    Wy = digital2analog(gyro[2, :], gyro_alpha_mean, gyro_beta[1])
    Wz = digital2analog(gyro[0, :], gyro_alpha_mean, gyro_beta[2])

    return Ax, Ay, Az, Wx, Wy, Wz


def get_q(x):
    q = Quaternion(x[0], x[1:4])
    q.normalize()
    return q


def get_w(x):
    return x[4:7]


def generate_sigma_points(x_k_minus_1, P_k_minus_1, Q):
    '''
    x_k_minus_1, 1x7 distribution mean
    P_k_minus_1, 6x6 distribution covariance matrix 
    Q, 6x6 the process noise

    generate 12 sigma points around the x_k_minus_1 

    return Xi, 7x12, every column is a sigma point
    '''
    n = 6

    # equation 35, 6x6, every column is a vector
    S = np.linalg.cholesky((P_k_minus_1 + Q))
    # equation 36, 6x12, every column is a sigma point
    W = np.sqrt(n/2) * np.hstack((S, -S))

    # the first 3 elements is q covariance
    # the last 3 elements is w covariance

    # should also be 7x12, every column is a sigma point
    Xi = np.zeros((7, 12))

    # equation 37, 7x12, every column is a sigma point
    q_k_minus_1 = get_q(x_k_minus_1)
    w_k_minus_1 = get_w(x_k_minus_1)
    for i in range(12):

        q_w = Quaternion()
        q_w.from_axis_angle(W[0:3, i])

        w_w = W[3:6, i]

        Xq = (q_k_minus_1 * q_w)
        Xq.normalize()
        Xi[0:4, i] = Xq.q

        Xi[4:7, i] = w_k_minus_1 + w_w

    return Xi


def get_q_delta(x, dt):
    w = get_w(x)
    q_delta = Quaternion()
    q_delta.from_axis_angle(w * dt)
    return q_delta


def process_model(Xi, dt):
    # do it for 12 points, no need to set random noise

    # equation 22, 7x12
    for i in range(12):
        # equation 11
        q_delta = get_q_delta(Xi[:, i], dt)
        # equation 12
        q_k = get_q(Xi[:, i])
        q_k_plus_1 = (q_k * q_delta)
        q_k_plus_1.normalize()
        Xi[0:4, i] = q_k_plus_1.q
        # Xi[4:7, i] does not change

    return Xi  # as Yi, 7x12


def calculate_quaternion_mean(qi, q_init):
    max_iterations = 10
    tolerance = 1e-2

    qt = Quaternion(q_init[0], q_init[1:4])
    qt.normalize()

    last_error = 100
    e_mean = np.array([0.1, 0.1, 0.1])
    for _ in range(max_iterations):

        error = np.linalg.norm(e_mean)

        if np.abs(error - last_error) < tolerance:
            break

        e = np.zeros((3, 12))
        for i in range(12):
            q = Quaternion(qi[0, i], qi[1:4, i])
            ei = q * qt.inv()
            ei.normalize()
            e[:, i] = ei.axis_angle()
        e_mean = np.mean(e, axis=1)
        q_e = Quaternion()
        q_e.from_axis_angle(e_mean)
        qt = q_e * qt
        qt.normalize()

    # print(qt.q)
    return qt.q


def calculate_mean(Yi):
    # Yi is 7x12
    x_k_bar = np.zeros(7)
    x_k_bar[0] = 1

    # angular velocity component directly mean
    x_k_bar[4:7] = np.mean(Yi[4:7, :], axis=1)
    # orientation component need gradient descent
    x_k_bar[0:4] = calculate_quaternion_mean(Yi[0:4, :], x_k_bar[0:4])
    return x_k_bar


def calculate_covariance(x_k_bar, Yi):
    # covariance matrix is 6x6

    n = 6

    # Wi = Xi - x_k_bar
    # W is 6x12, every column is a point
    W = np.zeros((6, 12))
    q_mean = Quaternion(x_k_bar[0], x_k_bar[1:4])
    w_mean = x_k_bar[4:7]

    for i in range(12):
        q_Y = Quaternion(Yi[0, i], Yi[1:4, i])
        # equation 67
        r_w = q_Y * q_mean.inv()
        r_w.normalize()
        W_q = r_w.axis_angle()
        W_w = Yi[4:7, i] - w_mean
        W[0:3, i] = W_q
        W[3:6, i] = W_w

    # equation 64
    return (W @ W.T)/2/n, W  # 6x6


def measurement_model(Y):
    Z = np.zeros((6, 12))
    # equation 27
    g = Quaternion(0.0, [0.0, 0.0, 9.8])

    for i in range(12):
        q_k = Quaternion(Y[0, i], Y[1:4, i])
        g_p = q_k.inv() * g * q_k
        # g_p.normalize()
        Z[0:3, i] = g_p.vec()
        Z[3:6, i] = Y[4:7, i]

    return Z


# roll, pitch, yaw = estimate_rot(data_num=1)
# # print(roll)
# # print(pitch[0:5])
# # print(yaw)


# # ploting the results
# # plt.plot(roll, label='roll')
# plt.plot(pitch, label='pitch')

# # plt.plot(yaw, label='yaw')
# plt.legend()
# plt.show()
