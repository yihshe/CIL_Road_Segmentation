"""
Road vector extraction from label image.
"""

import numpy as np
from graph import *

def extract_vectors(img):
    """
    Attempts to extract all potential road vectors from img.
    """
    border = border_points(img)
    vectors = []
    
    # Start at each border point
    for idx, p in enumerate(border):
        origin = p
        visited = set()
        
        # Follow road until we can
        while in_img(img, origin):
            visited.add(origin)
            best_candidate = None
            best_score = 0
            for c in candidates(img, origin, radius=3):
                score = along_score(img, origin, c, radius=1)

                if score > best_score:
                    best_score = score
                    best_candidate = c

            vectors.append([origin, best_candidate])
            origin = best_candidate

            # Break if starting to backtrack
            if origin in visited:
                break

    return vectors

def candidates(img, origin, radius):
    """
    Extract candidates around a given point.
    """
    candidates = []
    top, right = True, True

    # Always start by top candidate
    current = [origin[0] - radius, origin[1]]

    while True:
        if in_img(img, current):
            candidates.append((current[0], current[1]))

        # Generate next candidate
        current[0] = current[0] + 1 if right else current[0] - 1
        current[1] = current[1] + 1 if  top else current[1] - 1

        # Update orientation
        if current[0] == origin[0] + radius:
            right = False

        if (current[1] == origin[1] + radius):
            top = False
        elif (current[1] == origin[1] - radius):
            top = True

        # Return if loop completed
        if current == [origin[0] - radius, origin[1]]:
            return candidates

def along_score(img, origin, candidate, radius=1):
    """
    Compute how much does the vector follow a route (ignoring origin and end).
    """
    vector_path = path(img, origin, candidate)
    score = 0

    for p in vector_path[1:]:
        score = score + neighboring_score(img, p, radius)

    return score

def neighboring_score(img, point, radius=1):
    """
    Compute how well a given point is centered in a road.
    """
    score = 0

    for r in range(1, radius + 1):
        neighbors = neighbors_at_dist(img, point, r)

        for p in neighbors:
            if img[p[0], p[1]] == 0:
                score += 1

    return score

def filter_candidates(img, origin, candidates):
    """
    Return the best candidate vector or None if none can be a road.
    """
    for candidate in candidates:
        if img[candidate] == 0:
            return candidate
