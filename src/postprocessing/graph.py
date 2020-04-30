"""
Graph tools for road vectors extraction.
"""

import numpy as np
from scipy.sparse.csgraph import dijkstra

def graph(img, origin, candidate):
    """
    Compute the graph representing the grid between origin and candidate.

    The original grid
        1 2 3
        4 5 6
        7 8 9
    is transformed to its (9, 9) link matrix, with 1 the two nodes are linked
    """
    height, width = abs(candidate[0] - origin[0]) + 1, abs(candidate[1] - origin[1]) + 1
    g = np.zeros((height * width, height * width))

    # Shift to make origin the axis origin
    start = (0, 0)
    end = shift_origin(candidate, origin)

    for i in range(height):
        for j in range(width):
            p = coo_to_id((i, j), width)
            for neighbor in neighbors_at_dist(np.zeros((height, width)), (i, j), 1):
                g[p, coo_to_id(neighbor, width)] = 1

    return g

def border_points(img):
    """
    Get road pixel around the image.
    """
    height, width = img.shape[0], img.shape[1]

    up = [(0, i) for i in range(width)]
    left = [(i, 0) for i in range(height)]
    right = [(width - 1, i) for i in range(height)]
    down = [(height - 1, i) for i in range(width)]

    points = up
    points.extend(left)
    points.extend(right)
    points.extend(down)

    return list(filter(lambda x: img[x] == 0, points))

def path(img, origin, candidate):
    """
    Generate the pixels between origin and candidate following the shortest path.
    """
    height, width = abs(candidate[0] - origin[0]) + 1, abs(candidate[1] - origin[1]) + 1
    dists, preds = dijkstra(graph(img, origin, candidate), unweighted=True, return_predecessors=True)

    p = coo_to_id((candidate[0] - origin[0], candidate[1] - origin[1]), width)
    goal = 0
    path = [candidate]

    while p != goal:
        prev = preds[goal, p]
        subgrid_p = id_to_coo(prev, width)
        path.insert(0, (subgrid_p[0] + candidate[0], subgrid_p[1] + candidate[1]))
        p = prev

    return path

def neighbors_at_dist(img, p, dist):
    """
    Get neighbors of a given points a distance dist.
    Neighbors are represented by the square of side length dist.
    Example with dist = 1:
        - - -
        - x -
        - - -
    """
    # Start by top left corner
    current = (p[0] - dist, p[1] - dist)

    # Represents a square, side can be 0, 1, 2, 3, starting from top, clockwise
    side = 0
    neighbors = []

    while True:
        if in_img(img, current):
            neighbors.append(current)

        # Advance on square
        if side == 0:
            current = (current[0], current[1] + 1)
        elif side == 1:
            current = (current[0] + 1, current[1])
        elif side == 2:
            current = (current[0], current[1] - 1)
        else:
            current = (current[0] - 1, current[1])

        # Change direction when reaching corners
        if (current == (p[0] - dist, p[1] + dist) or
            current == (p[0] + dist, p[1] + dist) or
            current == (p[0] + dist, p[1] - dist)):
            side = side + 1

        # Stop when reaching back top left corner
        if current == (p[0] - dist, p[1] - dist):
            return neighbors

def in_img(img, point):
    """
    True if point is in img.
    """
    return (0 <= point[0] < img.shape[0]) and  (0 <= point[1] < img.shape[1])

def coo_to_id(point, width):
    """
    Transform a coordinate (x, y) to the corresponding node ID in a given grid.
    """
    return point[0] * width + point[1]

def id_to_coo(label, width):
    """
    Transform a node ID in a given grid to the corresponding coordinate (x, y).
    """
    return (int(label / width), label % width)

def shift_origin(point, origin):
    """
    Shift point with respect to origin.
    """
    return (point[0] - origin[0], point[1] - origin[1])
