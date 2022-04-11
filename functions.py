import math

import numpy as np

x0 = 1
y0 = 2
alpha = math.pi / 6


def f_xy(x: []) -> []:
    return (x[0] - math.sqrt(2)) ** 2 + (x[1] + math.sqrt(3)) ** 2


def grad_f_xy(x: []):
    return np.array([[2 * (float(x[0] - math.sqrt(2)))], [2 * (float(x[1] + math.sqrt(3)))]])


def h_matrix(x: []):
    return np.array([[2., 0.], [0., 2.]])


def f_cosh_xy(x: []) -> []:
    x_new = np.matrix([[(x[0, 0] - x0) * math.cos(alpha) + (x[1, 0] - y0) * math.sin(alpha)],
                       [(x[1, 0] - y0) * math.sin(alpha) - (x[0, 0] - x0) * math.cos(alpha)]], float)
    return math.cosh(x_new[0]) + math.cosh(x_new[1])


def grad_f_cosh_xy(x: []):
    x_new = np.matrix([[(x[0, 0] - x0) * math.cos(alpha) + (x[1, 0] - y0) * math.sin(alpha)],
                       [(x[1, 0] - y0) * math.sin(alpha) - (x[0, 0] - x0) * math.cos(alpha)]], float)
    return np.array([[math.sinh(x_new[0]) * math.cos(alpha)], [math.sinh(x_new[1]) * math.sin(alpha)]])


def h_cosh_xy(x: []):
    x_new = np.matrix([[(x[0, 0] - x0) * math.cos(alpha) + (x[1, 0] - y0) * math.sin(alpha)],
                       [(x[1, 0] - y0) * math.sin(alpha) - (x[0, 0] - x0) * math.cos(alpha)]], float)
    return np.array([[math.cosh(x_new[0]) * math.cos(alpha) ** 2, 0.], [0., math.cosh(x[1]) * math.sin(alpha) ** 2]])
