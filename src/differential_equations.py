#!/usr/bin/env python
import numpy as np


class DifferentialEquations:
    def __init__(self, *args):

        self.args = args

    def runge_kutta_order_4(self, delta_time, i_th_iteration):
        """
        Calculation of i+1 iteration using range-kutta method.

        This calculating next iteration from values of current iteration
        with help of runge-kutta method of order 4.

        Parameters
        ----------
        delta_time : float
            time step between two time iterations.
            delta time is the change in independent variable
            between 2 iterations
        i_th_iteration : array
            values of independent and dependent variables for current iteration
            ith iteration matrix has time in the 0th index and x in 1st index
            and dx/dt in 2nd index and so on.

        """

        # intializing runge kutta constants to be zero matrices.Constant
        # matrices are defined as [ K1 L1 M1 ....... ]
        # Length of runge kutta constants length
        range_kutta_constant_length = len(i_th_iteration)-1
        range_kutta_constant_1 = np.zeros(range_kutta_constant_length)
        range_kutta_constant_2 = np.zeros(range_kutta_constant_length)
        range_kutta_constant_3 = np.zeros(range_kutta_constant_length)
        range_kutta_constant_4 = np.zeros(range_kutta_constant_length)

        # Initialization of i_plus_1_th_iteration matrix
        i_plus_1_th_iteration = np.zeros(len(i_th_iteration))

        # This dummy variable is defined to add i_th_iteration matrix and
        # range kutta constants as the both have different lengths
        dummy_argument = np.zeros(len(i_th_iteration))

        # Calculating 1st range kutta constant
        for index, arg in enumerate(self.args):
            range_kutta_constant_1[index] = delta_time*arg(i_th_iteration)

        # Calculating 2nd range kutta constant
        for index, arg in enumerate(self.args):
            dummy_argument[0] = i_th_iteration[0]+delta_time/2
            dummy_argument[1:] = i_th_iteration[1:]+range_kutta_constant_1/2

            range_kutta_constant_2[index] = delta_time*arg(dummy_argument)

        # Calculating 3rd range kutta constant
        for index, arg in enumerate(self.args):
            dummy_argument[0] = i_th_iteration[0]+delta_time/2
            dummy_argument[1:] = i_th_iteration[1:]+range_kutta_constant_2/2

            range_kutta_constant_3[index] = delta_time*arg(dummy_argument)

        # Calculating 4th range kutta constant
        for index, arg in enumerate(self.args):
            dummy_argument[0] = i_th_iteration[0]+delta_time
            dummy_argument[1:] = i_th_iteration[1:]+range_kutta_constant_3

            range_kutta_constant_4[index] = delta_time*arg(dummy_argument)
        # i+1 iteration = i iteration + (K1+2*K2+2*K3+K4)/6
        # value at index 0 for time is handled appropriately
        i_plus_1_th_iteration[0] = i_th_iteration[0]+delta_time

        i_plus_1_th_iteration[1:] = i_th_iteration[1:] \
            + (range_kutta_constant_1+2*range_kutta_constant_2
               + 2*range_kutta_constant_3+range_kutta_constant_4)/6

        return i_plus_1_th_iteration

    def solution_differential_eqn(self, delta_time, iter_num, init_cond):
        """
        Solving differential equation using runge kutta method of order 4.

        The final solution matrix has each row representing each
        iteration. The column represent variables. The zeroth column is
        time, first is x, second is dx/dt.

        Parameters
        ----------
        delta_time : float
            time step between two time iterations.
        iter_num : int
            total number of iterations to be performed.
        init_cond : array
            initial conditions for the differential equation.
        """

        # starting point for 1st iteration
        i_th_iteration = init_cond
        # solution matrix(rows represent iterations and columns represent variable.
        solution = np.zeros([iter_num, len(i_th_iteration)])
        # 1st row of solution matrix is set equal to initial conditions
        solution[0, :] = init_cond

        for i in range(1, iter_num):
            # initial_time=i_th_iteration[0]
            i_th_iteration = self.runge_kutta_order_4(
                delta_time, i_th_iteration)
            solution[i, :] = i_th_iteration
        return solution
