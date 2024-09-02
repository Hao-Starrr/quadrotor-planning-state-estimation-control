import numpy as np

from .graph_search import graph_search


class WorldTraj(object):
    """

    """

    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.5

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(
            world, self.resolution, self.margin, start, goal, astar=True)
        print("Path found:", self.path)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.

        # every 10 points in the path, add the end point
        # self.points = self.path[::10]  # shape=(n_pts,3)
        # self.points = np.vstack((self.points, self.path[-1]))
        self.points = self.resample_by_distance(
            self.path, distance_threshold=1.48)
        # Ramer–Douglas–Peucker algorithm is better
        # self.points = self.douglas_peucker(self.path, 0.4)

        self.num_points = self.points.shape[0]

        # segments
        # [i] segment is from points[i] to points[i+1]
        self.num_segments = self.num_points - 1
        self.direction = (self.points[1:] - self.points[:-1])  # not normalized

        # speed
        self.speed = 2.0  # m/s

        # duration
        self.duration = np.zeros((self.num_segments))
        for i in range(self.num_segments):
            distance = np.linalg.norm(self.direction[i])
            self.duration[i] = distance / self.speed
            # if distance < 0.3:
            #     self.duration[i] *= 1.2
            # elif distance > 2.0:
            #     self.duration[i] *= 0.8

        # trick, slow down the last segment twice as fast
        # self.duration[-2] *= 1.3
        self.duration[-1] *= 2.0
        self.duration[0] *= 1.5

        # timestamps
        # [i] segment is from timestamps[i] to timestamps[i+1]
        self.timestamps = np.hstack((0, np.cumsum(self.duration)))

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

        # build the matrix of coefficients for all
        # one segment will have 6 coefficients
        self.A = np.zeros((self.num_segments * 6, self.num_segments * 6))

        # the fist 6 rows are always start and end point, x v a
        self.A[0:3, 0:6] = self.first_to_third_order_coefficient(0)
        self.A[3:6, -
               6:] = self.first_to_third_order_coefficient(self.duration[-1])
        for i in range(1, self.num_segments):
            self.A[6*i:6*(i+1), 6*(i-1):6*(i+1)
                   ] = self.one_waypoint_coefficient(self.duration[i-1])

        self.b = np.zeros((self.num_segments * 6, 3))

        # start point
        self.b[0, :] = self.points[0]
        self.b[1, :] = np.zeros(3)
        self.b[2, :] = np.zeros(3)
        # end point
        self.b[3, :] = self.points[-1]
        self.b[4, :] = np.zeros(3)
        self.b[5, :] = np.zeros(3)
        # for every point in the middle
        for i in range(1, self.num_segments):
            self.b[6*i, :] = self.points[i]
            self.b[6*i+1, :] = self.points[i]

        # solve the linear system
        # x = np.linalg.solve(A, b)
        # 6n * 3
        self.coefficient = np.linalg.solve(self.A, self.b)

        # in ceofficients, every 6 rows are for one segment

        # for example, first_order_coefficient(t) @ c1 = segment 1 position
        # second_order_coefficient(t) @ c2 = segment 2 velocity

    def resample_by_distance(self, path, distance_threshold=1.5):
        sampled_points = [path[0]]
        last_added_point = path[0]
        for point in path:
            if np.linalg.norm(point - last_added_point) > distance_threshold:
                sampled_points.append(point)
                last_added_point = point
        if not np.array_equal(sampled_points[-1], path[-1]):
            sampled_points.append(path[-1])

        return np.array(sampled_points)

    def perpendicular_distance(self, point, line_start, line_end):
        line_vector = line_end - line_start
        line_direction = line_vector / np.linalg.norm(line_vector)
        point_vec = point - line_start
        projection_length = np.dot(point_vec, line_direction)
        nearest = line_start + projection_length * line_direction
        return np.linalg.norm(point - nearest)

    def douglas_peucker(self, path, epsilon):
        dmax = 0
        index = -1
        end = len(path) - 1
        for i in range(1, end):
            d = self.perpendicular_distance(
                path[i], path[0], path[end])
            if d > dmax:
                index = i
                dmax = d

        if dmax > epsilon:
            rec_results1 = self.douglas_peucker(path[:index+1], epsilon)
            rec_results2 = self.douglas_peucker(path[index:], epsilon)

            points = np.vstack((rec_results1[:-1], rec_results2))
        else:
            points = np.array([path[0], path[end]])

        return points

    def first_order_coefficient(self, t):
        return np.array([t**5, t**4, t**3, t**2, t**1, 1])

    def second_order_coefficient(self, t):
        return np.array([5*t**4, 4*t**3, 3*t**2, 2*t, 1, 0])

    def third_order_coefficient(self, t):
        return np.array([20*t**3, 12*t**2, 6*t, 2, 0, 0])

    def fourth_order_coefficient(self, t):
        return np.array([60*t**2, 24*t, 6, 0, 0, 0])

    def fifth_order_coefficient(self, t):
        return np.array([120*t, 24, 0, 0, 0, 0])

    def first_to_third_order_coefficient(self, t):  # 3x6
        return np.vstack((self.first_order_coefficient(t), self.second_order_coefficient(t), self.third_order_coefficient(t)))

    def intermediates_portion(self, t):  # 2x12
        left_up = self.first_order_coefficient(t)
        left_down = np.zeros(6)
        left = np.vstack((left_up, left_down))
        right_up = np.zeros(6)
        right_down = self.first_order_coefficient(0)
        right = np.vstack((right_up, right_down))
        return np.hstack((left, right))

    def continuity_portion(self, t):  # 4x12
        left = np.vstack((self.second_order_coefficient(t), self.third_order_coefficient(
            t), self.fourth_order_coefficient(t), self.fifth_order_coefficient(t)))
        right = np.vstack((self.second_order_coefficient(0), self.third_order_coefficient(
            0), self.fourth_order_coefficient(0), self.fifth_order_coefficient(0)))
        right *= -1
        return np.hstack((left, right))

    def one_waypoint_coefficient(self, t):  # 6x12
        return np.vstack((self.intermediates_portion(t), self.continuity_portion(t)))

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE

        if t >= self.timestamps[-1]:
            x = self.points[-1]
            # x_dot = self.direction[-1]
            flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                           'yaw': yaw, 'yaw_dot': yaw_dot}
            return flat_output

        else:
            # else find the segment
            i = 0
            while t >= self.timestamps[i]:
                i += 1
            # t will be in timestamp[i-1] and timestamp[i]
            # which is in segment i-1

            # 6x3
            segment_coefficient = self.coefficient[6*(i-1):6*i, :]

            segment_time = t - self.timestamps[i-1]

            x, x_dot, x_ddot, x_dddot, x_ddddot = self.calculate_x(
                segment_time, segment_coefficient)

            # constant speed
            # x = self.points[i-1] + (t - self.timestamps[i-1]) * \
            #     self.direction[i-1] * self.speed / \
            #     np.linalg.norm(self.direction[i-1])
            # x_dot = self.direction[i-1] * \
            #     self.speed / np.linalg.norm(self.direction[i-1])

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output

    def calculate_x(self, segment_time, segment_coefficient):
        x = (self.first_order_coefficient(segment_time)
             @ segment_coefficient).flatten()
        x_dot = (self.second_order_coefficient(
            segment_time) @ segment_coefficient).flatten()
        x_ddot = (self.third_order_coefficient(segment_time)
                  @ segment_coefficient).flatten()
        x_dddot = (self.fourth_order_coefficient(
            segment_time) @ segment_coefficient).flatten()
        x_ddddot = (self.fifth_order_coefficient(
            segment_time) @ segment_coefficient).flatten()
        return x, x_dot, x_ddot, x_dddot, x_ddddot
