from heapq import heappush, heappop  # Recommended.
import numpy as np
import math

from flightsim.world import World

from .occupancy_map import OccupancyMap  # Recommended.


# math.sqrt better than np.sqrt
# tie break
# heap maintain (how to deal with repeated nodes)
# diagonal heuristic


def graph_search(world, resolution, margin, start, goal, astar):
    """
    Parameters:
        world,      World object representing the environment obstacles
        resolution, xyz resolution in meters for an occupancy map, shape=(3,)
        margin,     minimum allowed distance in meters from path to obstacles.
        start,      xyz position in meters, shape=(3,)
        goal,       xyz position in meters, shape=(3,)
        astar,      if True use A*, else use Dijkstra
    Output:
        return a tuple (path, nodes_expanded)
        path,       xyz position coordinates along the path in meters with
                    shape=(N,3). These are typically the centers of visited
                    voxels of an occupancy map. The first point must be the
                    start and the last point must be the goal. If no path
                    exists, return None.
        nodes_expanded, the number of nodes that have been expanded
    """

    # While not required, we have provided an occupancy map you may use or modify.
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))
    print("start_index:", start_index)
    print("goal_index:", goal_index)

    # ini
    open_list = []  # store nodes to be explored (index)
    close_list = set()  # store explored nodes (index)
    parent = {}
    nodes_expanded = 0

    g_score = {start_index: 0}
    f_score = {start_index: heuristic(start_index, goal_index) if astar else 0}
    heappush(open_list, (f_score[start_index], start_index))

    # main loop
    while open_list:
        # remove first node from open_list
        current_f, current_index = heappop(open_list)

        # if it is the goal, then we have found the path
        if current_index == goal_index:
            path = reconstruct_path(
                occ_map, parent, current_index, start, goal)
            # print(np.array(path))
            return np.array(path), nodes_expanded

        close_list.add(current_index)
        nodes_expanded += 1

        # add the neighbors to the open_list
        for neighbor in get_expandable_neighbors(occ_map, current_index):
            # case 1: neighbor is already in close_list
            # do nothing
            if neighbor in close_list:
                continue

            potential_g_score = g_score[current_index] + \
                heuristic(current_index, neighbor)
            potential_f_score = potential_g_score + \
                (heuristic(neighbor, goal_index) if astar else 0)

            # case 2: neighbor is not in open_list
            # directly add it to the open_list

            # case 3: neighbor is in open_list
            # check if it has a lower f_cost
            # if yes, add it to the open_list with the new f
            if neighbor not in f_score or potential_f_score < f_score[neighbor]:
                parent[neighbor] = current_index
                g_score[neighbor] = potential_g_score
                f_score[neighbor] = potential_f_score
                if neighbor not in [item[1] for item in open_list]:
                    heappush(open_list, (f_score[neighbor], neighbor))
                else:
                    # it should update the f_score and re-sort the heap
                    pass

    # if no path
    return None, nodes_expanded


def heuristic(a, b):
    # use Euclidean distance as the heuristic
    h = a[0]-b[0], a[1]-b[1], a[2]-b[2]
    return math.sqrt(h[0]**2 + h[1]**2 + h[2]**2)
    # return np.linalg.norm(np.array(a) - np.array(b))


def is_expandable(occ_map, index):
    # is is within the bounds and not occupied
    return occ_map.is_valid_index(index) and not occ_map.is_occupied_index(index)


def get_expandable_neighbors(occ_map, index):
    # will return a list of valid neighbors of the given index
    i, j, k = index
    neighbors = []
    # inverse the direction of the neighbor to make it expand to larger index first
    for di in [1, 0, -1]:
        for dj in [1, 0, -1]:
            for dk in [1, 0, -1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue
                neighbor_index = (i + di, j + dj, k + dk)
                if is_expandable(occ_map, neighbor_index):
                    neighbors.append(neighbor_index)
    return neighbors


def reconstruct_path(occ_map, parent, current, start, goal):
    total_path = [goal]

    current_posi = tuple(occ_map.index_to_metric_center(current))
    total_path.append(current_posi)

    while current in parent:
        current = parent[current]
        current_posi = tuple(occ_map.index_to_metric_center(current))
        total_path.append(current_posi)

    total_path.append(start)

    total_path.reverse()
    # print("total_path:", total_path)
    return total_path
