import math
import numpy as np


def point_line_dist(p1, p2, p3):
    """
    Calculate the distance and a projection point between a point and a line.

    Parameters
    ==========

    p1, p2 : points for the line

    p3 : point

    Returns
    =======

    p : np.ndarray
        Projection of p3 to p1p2

    d : float
        Distance from p3 to p1p2.

    Examples
    ========

    >>> from mcplexpt.util import point_line_dist
    >>> p1, p2, p3 = [-1, 0], [1, 0], [0, 1]
    >>> point_line_dist(p1, p2, p3)
    (array([0., 0.]), 1.0)

    """
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    u12, u13 = p2 - p1, p3 - p1
    t = (np.dot(u13, u12)/np.linalg.norm(u12)**2)
    p = p1 + t*u12
    d = np.linalg.norm(p3 - p)
    return p, d


def halve_angle(p1, p2, p3):
    """
    Calculate the line which halves the angle formed by *p1*, *p2*, and *p3*.

    Parameters
    ==========

    p1, p2, p3 : points

    Returns
    =======

    k : float
        Slope of the line.

    b : float
        y-intercept of the line

    Examples
    ========

    >>> from mcplexpt.util import halve_angle
    >>> p1, p2, p3 = [1, 2], [0, 2], [0, 4]
    >>> k, b = halve_angle(p1, p2, p3)
    >>> round(k, 2)
    1.0
    >>> round(b, 2)
    2.0

    """
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v21, v23 = p1 - p2, p3 - p2
    theta_v21 = math.atan2(*v21)
    theta_v23 = math.atan2(*v23)
    theta_half = (theta_v21 + theta_v23)/2
    k = math.tan(theta_half)
    b = p2[1] - k*p2[0]
    return k, b
