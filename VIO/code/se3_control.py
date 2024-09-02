import numpy as np
from scipy.spatial.transform import Rotation


class SE3Control(object):
    """

    """

    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.arm_length = quad_params['arm_length']  # meters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag = quad_params['k_drag']   # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(
            np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2

        # STUDENT CODE HERE
        self.Kp = 5.58 * np.eye(3)
        self.Kd = 3.30 * np.eye(3)
        self.KR = 270 * np.eye(3)
        self.Kw = 25 * np.eye(3)

        l = self.arm_length
        ga = self.k_drag / self.k_thrust
        coefficent_matrix = np.array(
            [[1, 1, 1, 1],
             [0, l, 0, -l],
             [-l, 0, l, 0],
             [ga, -ga, ga, -ga]])
        self.inv_coefficent_matrix = np.linalg.inv(coefficent_matrix)

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE

        # PID calculate desired acceleration
        r_ddot_des = flat_output['x_ddot']
        r_ddot_des -= self.Kd @ (state['v'] - flat_output['x_dot'])
        r_ddot_des -= self.Kp @ (state['x'] - flat_output['x'])

        # desired force direction = desired acceleration direction
        gravity = np.array([0, 0, self.mass * self.g])
        F_des = self.mass * r_ddot_des + gravity
        b3_des = F_des / np.linalg.norm(F_des)

        phi = flat_output['yaw']
        a_phi = np.array([np.cos(phi), np.sin(phi), 0])
        b2_des = np.cross(b3_des, a_phi, axis=0)
        b2_des = b2_des / np.linalg.norm(b2_des)

        b1_des = np.cross(b2_des, b3_des, axis=0)
        R_des = np.vstack([b1_des, b2_des, b3_des]).T

        # current rotation matrix and direction
        R = Rotation.from_quat(state['q']).as_matrix()
        b3 = R[:, 2]

        # calculate orientation error
        error_matrix = (R_des.T @ R - R.T @ R_des) / 2
        # error vector
        e_R = np.array(
            [error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])

        # u1
        u = np.zeros(4)
        u[0] = np.dot(b3, F_des)

        # u2
        e_omega = state['w']  # no desired omega
        u[1:] = (self.inertia @ (-self.KR @ e_R - self.Kw @ e_omega))

        F = self.inv_coefficent_matrix @ u

        F[F < 0] = 0

        cmd_motor_speeds[:] = np.sqrt(F/self.k_thrust).reshape((4,))
        # cmd_thrust = F.sum()
        # cmd_moment[:] = u[1:].reshape((3,))
        cmd_q = Rotation.from_matrix(R_des).as_quat()

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q}
        return control_input
