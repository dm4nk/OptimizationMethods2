import math
import numpy

from functions import grad_f_xy, f_xy, h_matrix


def coordinate_descent(f, x, step, epsilon):
    """
    Coordinate Descent
    :param f: function
    :param x: starting points
    :param step: step used to calculate coordinates
    :param epsilon: vector of error
    :return: minimum of function
    """
    iteration = 0
    while True:
        x_prev = numpy.copy(x)
        for i in range(0, len(x)):
            x_new = numpy.copy(x)
            x_new[i] += step
            fx = f(x)
            fx_new = f(x_new)

            while fx_new < fx:
                x[i] = x_new[i]
                fx = fx_new
                x_new[i] += step
                fx_new = f(x_new)
                iteration += 1

        if all(numpy.abs(x - x_prev) < epsilon):
            print(f"Iterations: {iteration}")
            return x


def gradient_descent(f, grad_f, x, step, epsilon):
    """
    Gradient Descent
    :param step: step used to calculate coordinates
    :param epsilon: vector of error
    :param x: starting points
    :param grad_f: gradient of function
    :param f: function
    :return: minimum of function
    """
    iteration = 0
    while True:
        x_prev = numpy.copy(x)
        fx_prev = f(x_prev)
        grad_fx = grad_f(x)
        x -= step * grad_fx
        fx = f(x)
        while fx < fx_prev:
            x -= step * grad_fx
            fx_prev = fx
            fx = f(x)
            iteration += 1

        if all(numpy.abs(x - x_prev) < epsilon):
            print(f"Iterations: {iteration}")
            return x


def newton_descent(f, grad_f, H, x, epsilon):
    """
    Newton Descent
    :param f: function
    :param grad_f: gradient of f
    :param H: matrix of second derivatives of f
    :param x: starting points
    :param epsilon: vector of error
    :return: minimum of function
    :return: minimum of function
    """
    iteration = 0
    while True:
        x_prev = numpy.copy(x)
        x = x - numpy.linalg.inv(H(x)).dot(grad_f(x))
        iteration += 1

        if all(numpy.abs(x - x_prev) < epsilon):
            print(f"Iterations: {iteration}")
            return x


def main():
    eps = 0.01
    step = 0.005
    eps_mas = numpy.array([[eps], [eps]])
    x = numpy.matrix([[-1.], [-2.]])
    print("coordinate_descent min: ", coordinate_descent(f_xy, x, step, eps_mas), "\n", "-" * 40)
    print("gradient_descent min: ", gradient_descent(f_xy, grad_f_xy, x, step, eps_mas), "\n", "-" * 40)
    print("newton_descent min: ", newton_descent(f_xy, grad_f_xy, h_matrix, x, eps_mas), "\n", "-" * 40)


if __name__ == "__main__":
    main()
