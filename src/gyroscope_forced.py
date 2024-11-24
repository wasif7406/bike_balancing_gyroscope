#!/usr/bin/env python

import os
import matplotlib.pyplot as plt
import numpy as np
import differential_equations

# All units are in SI with exception to degrees which are used 
# in plotting graphs.
# Gyroscope parameters
largest_gyro_length = 11.4
rod_rad = 1 * 10**-2
disk_rad = np.sqrt(4 / 5) * largest_gyro_length * 10**-2
rod_len = np.sqrt(4 / 5) * largest_gyro_length * 10**-2
# rod_len = 10*10**-2
disk_len = rod_len
# disk_pos = 6.5*10**-2
disk_pos = 0
density = 7.85 * 10**3

# mass of gyroscope calculations
# gyro_mass = density*np.pi*(rod_rad**2*rod_len+(disk_rad**2-rod_rad**2)*disk_len)
gyro_mass = 6.2155

# center of mass of gyroscope calculations
# x_mass_centre = (density*np.pi/gyro_mass)*(rod_rad**2*rod_len**2/2+(disk_rad**2 \
#        -rod_rad**2)*disk_len*(disk_pos+disk_len/2))
#
# Moment of inertia of gyroscope about endpoint of shaft
# I_endpoint = density*np.pi*(rod_rad**4*rod_len/4+((disk_rad**4-rod_rad**4) \
#        *disk_len/4)+(rod_rad**2*rod_len**3)/3+(disk_rad**2-rod_rad**2) \
#        *(disk_len**3+3*disk_pos**2*disk_len+3*disk_len**2*disk_pos)/3)

# Moment of inertia of gyroscope about center of mass using parallel axis theorem
# This moment of inertia is about both alpha and zeta axis.
# I_xi_xi = I_endpoint - gyro_mass*x_mass_centre**2
I_xi_xi = 0.03509464

# This moment of inertia is about both phi axis.
# I_phi_phi = (1/2)*density*np.pi*(disk_len*disk_rad**4-(rod_len-disk_len)*rod_rad**4)
I_phi_phi = 0.067557

# mass of bike(100kg) plus addition payload such as weight of people and other things.
bike_mass = 180
grav_accel = 9.81
# center of mass of bike plus payload just choosen to be at saddle of bike as
# center of mass will be around that position
# Just a rough approximation
bike_mass_center = 0.3

# The position where gyroscope is pivoted. This position is assigned relative to
# the ground.
gyro_pos = 0.45
# position where pivot is located on the shaft relative to endpoint of shaft of
# gyroscope
pivot_pos = 1 * 10**-2
# length from pivot to gyroscope center of mass
# pivot_len = x_mass_centre-pivot_pos
pivot_len = 0
# angular velocity of gyroscope about phi axis in radian/s
gyro_vel = 500
# number of iterations
iter_num = 30000
# moment of inertia of bike plus that of payload
# estimated as modeling people as bars
# Calculated bike moment of inertia using cading software and finding
# approximate radius of gyration.
I_bike = 30

# intial conditions
# difference between time interval of individual iterations
time_diff = 0.001
# Starting time
init_time = 0
# electrical constants associated with side motor circuit
motor_torq_const = 1.875
back_emf_const = 0.096
inductance = 0.000119
resistance = 0.61
# intial condition matrix 1st entry is that of independent variable latter
# corresponds to dependent variables
init_cond = np.array([init_time, 0.31, 0, 0])
# conjugate momentum in phi
momentum_phi = I_phi_phi * (gyro_vel + init_cond[2] * np.sin(init_cond[3]))

# constants
oscillation_const = (((bike_mass * grav_accel * bike_mass_center) +
                      (2 * gyro_mass * grav_accel * gyro_pos)) /
                     (2 * I_phi_phi * gyro_vel)) + 15

damping_const = (np.sqrt(
    (I_bike + 2 * gyro_mass * gyro_pos**2) *
    ((2 * I_phi_phi * gyro_vel) * (oscillation_const) -
     (bike_mass * grav_accel * bike_mass_center +
      2 * gyro_mass * grav_accel * gyro_pos))) / (I_phi_phi + gyro_vel)) + 5

proportional_const = 1.8

# functions corresponding to differential equations


def del_theta(x_input):
    """
    This function corresponds to derivative of theta.

    During solutions of differential equations using runge-kutta method
    of order 4. Second order differential equations have to be converted to
    2 first order differential equations. This function corresponds to
    derivative of theta. Theta is the angle of bike from the vertical
    to the ground.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    theta_dot = x_input[2]
    return theta_dot


def del_theta_dot(x_input):
    """
    This function corresponds to second derivative of theta.

    During solutions of differential equations using runge-kutta method
    of order 4. Second order differential equations have to be converted to
    2 first order differential equations. This function corresponds to
    second derivative of theta. Theta is the angle of bike from
    the vertical to the ground.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    # This if elif statement is used as to allow both one and
    # two dimension array input, as two dimension array computation
    # is useful in graphs
    if x_input.ndim == 1:
        theta = x_input[1]
        theta_dot = x_input[2]
        alpha = x_input[3]
        alpha_dot = del_alpha(x_input)
    elif x_input.ndim == 2:
        theta = x_input[:, 1]
        theta_dot = x_input[:, 2]
        alpha = x_input[:, 3]
        alpha_dot = del_alpha(x_input)

    bike_weight_torq = bike_mass * grav_accel * bike_mass_center * np.sin(
        theta)
    gyro_weight_torq = 2*gyro_mass*grav_accel*(gyro_pos
                                               + pivot_len*np.cos(alpha)) \
        * np.sin(theta)

    restoring_torq = 2 * momentum_phi * np.cos(alpha) * alpha_dot
    other_torq = 4 * (I_xi_xi*np.cos(alpha)+gyro_mass
                      * (gyro_pos+pivot_len*np.cos(alpha))*pivot_len) \
        * theta_dot*alpha_dot*np.sin(alpha)

    effec_inertia = I_bike+2*gyro_mass*(gyro_pos+pivot_len*np.cos(alpha))**2 \
        + 2*I_xi_xi*(np.cos(alpha))**2

    theta_dot_dot = (bike_weight_torq + gyro_weight_torq - restoring_torq +
                     other_torq) / effec_inertia

    return theta_dot_dot


def del_alpha(x_input):
    """
    This function corresponds to derivative of alpha.

    During solutions of differential equations using runge-kutta method
    of order 4. Second order differential equations have to be converted to
    2 first order differential equations. This function corresponds to
    derivative of alpha. Alpha is the angle of gyroscope with respect to
    the normal to the bike seat. This normal is oriented by angle theta
    from the vertical to the ground.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    # This if elif statement is used as to allow both one and
    # two dimension array input, as two dimension array computation
    # is useful in graphs
    if x_input.ndim == 1:
        theta = x_input[1]
        theta_dot = x_input[2]
        alpha = x_input[3]
    elif x_input.ndim == 2:
        theta = x_input[:, 1]
        theta_dot = x_input[:, 2]
        alpha = x_input[:, 3]

#    alpha_dot =(oscillation_const*np.sin(theta)+damping_const*theta_dot \
#            +proportional_const*alpha)/np.cos(alpha)
    alpha_dot = (oscillation_const * np.sin(theta) +
                 damping_const * theta_dot + proportional_const * alpha)
    return alpha_dot


# solution of differential equations
eqn = differential_equations.DifferentialEquations(del_theta, del_theta_dot,
                                                   del_alpha)

# solution matrix
solution_matrix = eqn.solution_differential_eqn(time_diff, iter_num, init_cond)

# this function corresponds to torque calculations
theta_dot_dot_mat = del_theta_dot(solution_matrix)
alpha_dot_mat = del_alpha(solution_matrix)
momentum_phi_mat = momentum_phi * np.ones(len(solution_matrix[:, 0]))


def del_alpha_dot(x_input):
    """
    This function corresponds to second derivative of alpha.

    Alpha is the angle of gyroscope with respect to the normal to the bike
    seat. This normal is oriented by angle theta from the vertical to the
    ground. This function is calculating the value of second derivative of
    alpha.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    theta = x_input[:, 1]
    theta_dot = x_input[:, 2]
    theta_dot_dot = theta_dot_dot_mat
    alpha_dot = alpha_dot_mat

    #    alpha_dot_dot=(oscillation_const*np.cos(theta)*theta_dot \
    #            +damping_const*theta_dot_dot+proportional_const*alpha_dot \
    #            +(alpha_dot**2)*np.sin(alpha))/np.cos(alpha)

    alpha_dot_dot = (oscillation_const * np.cos(theta) * theta_dot +
                     damping_const * theta_dot_dot +
                     proportional_const * alpha_dot)
    return alpha_dot_dot


def del_phi_dot(x_input):
    """
    This function corresponds to second derivative of phi.

    Phi is the rotation angle of gyroscope about its axis. This function is
    calculating the value of second derivative of phi.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    theta_dot = x_input[:, 2]
    theta_dot_dot = theta_dot_dot_mat
    alpha = x_input[:, 3]
    alpha_dot = alpha_dot_mat

    phi_dot_dot = - theta_dot_dot \
        * np.sin(alpha)-theta_dot*alpha_dot*np.cos(alpha)
    return phi_dot_dot


alpha_dot_dot_mat = del_alpha_dot(solution_matrix)


def torque(x_input):
    """
    This function finds the required torque on the gyroscope.

    This function is defined to calculate the necessary torque acting on the
    gyroscopes to balance the bike.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    theta = x_input[:, 1]
    theta_dot = x_input[:, 2]
    alpha = x_input[:, 3]
    alpha_dot_dot = alpha_dot_dot_mat

    mass_torq = gyro_mass*pivot_len*(gyro_pos+pivot_len*np.cos(alpha)) \
        * np.sin(alpha)*theta_dot**2

    restoring_torq = momentum_phi * theta_dot * np.cos(alpha)
    inertia_torq = I_xi_xi * theta_dot**2 * np.cos(alpha) * np.sin(alpha)
    weight_torq = gyro_mass*grav_accel*pivot_len*np.sin(alpha) \
        * np.cos(theta)

    effec_inertia = gyro_mass * pivot_len**2 + I_xi_xi

    torque_mat = effec_inertia*alpha_dot_dot \
        - (-mass_torq+restoring_torq-inertia_torq+weight_torq)
    return torque_mat


torque_mat = torque(solution_matrix)


def energy(x_input):
    """
    This is energy function of the system.

    This function is defined to calculate the total energy of the
    bike-gyroscope system.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    theta = x_input[:, 1]
    theta_dot = x_input[:, 2]
    alpha = x_input[:, 3]
    alpha_dot = alpha_dot_mat
    momentum_phi = momentum_phi_mat

    energy_mat = (1/2)*I_bike*theta_dot**2 \
        + gyro_mass*(((gyro_pos+pivot_len*np.cos(alpha))**2)*theta_dot**2) \
        + I_xi_xi*(np.cos(alpha)*theta_dot)**2 \
        + (gyro_mass*pivot_len**2+I_xi_xi)*(alpha_dot**2) \
        + (momentum_phi**2/I_phi_phi) \
        + bike_mass*grav_accel*bike_mass_center*np.cos(theta) \
        + 2*gyro_mass*grav_accel*(gyro_pos+pivot_len *
                                  np.cos(alpha))*np.cos(theta)
    return energy_mat


def phi_dot(x_input):
    """
    This function corresponds to derivative of phi.

    Phi is the rotation angle of gyroscope about its axis. This function is
    calculating the value of derivative of phi.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    theta_dot = x_input[:, 2]
    alpha = x_input[:, 3]
    momentum_phi = momentum_phi_mat
    phi_dot_mat = (momentum_phi / I_phi_phi) - theta_dot * np.sin(alpha)
    return phi_dot_mat


def momentum_theta(x_input):
    """
    This function corresponds to conjugate momentum for theta.

    Conjugate momentum is the generalization of momentum for
    generalized coordinates. As theta is angle so this will
    corresponds to angular momentum corresponding to theta.
    Theta is the angle of bike from the vertical to the ground.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    theta_dot = x_input[:, 2]
    alpha = x_input[:, 3]

    momentum_theta_mat = I_bike*theta_dot \
        + 2*gyro_mass*((gyro_pos+pivot_len*np.cos(alpha))**2)*theta_dot \
        + 2*momentum_phi*np.sin(alpha)+2*I_xi_xi \
        * ((np.cos(alpha))**2)*theta_dot

    return momentum_theta_mat


def momentum_alpha():
    """
    This function corresponds to conjugate momentum for alpha.

    Conjugate momentum is the generalization of momentum for generalized
    coordinates. As alpha is angle so this will corresponds to angular momentum
    corresponding to alpha.  Alpha is the angle of gyroscope with respect to
    the normal to the bike seat. This normal is oriented by angle theta from
    the vertical to the ground.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    alpha_dot = alpha_dot_mat
    return (I_xi_xi + gyro_mass * pivot_len**2) * (alpha_dot)


def del_energy(x_input):
    """
    This function corresponds to 2*integral of torque wrt alpha.

    This function is defined to calculate the change in energy of the
    bike-gyroscope system.

    Parameters
    ----------
    x_input : array
        These are the input variables and their derivatives stored in the
        array. The value at index zero is time(independent variable) while
        at index 1 is 1st dependent variable followed by the its derivative
        at index 2. Similarly for other dependent variable.
    """
    return 2 * np.sum(torque_mat[:-1] * (x_input[1:, 3] - x_input[:-1, 3]))


current = torque_mat / motor_torq_const
del_current = np.zeros(len(current))
del_current[:-1] = (current[1:] - current[:-1]) / time_diff
voltage = inductance*del_current + resistance \
    * current + back_emf_const*alpha_dot_mat
# print(del_energy(solution_matrix))
# print(energy(solution_matrix)[iter_num-1] - energy(solution_matrix)[0])

try:
    os.mkdir("gyroscope_forced_graphs")
except:
    pass

try:
    os.chdir('gyroscope_forced_graphs')
except:
    exit()
# plot for theta vs time
plt.figure(1)
plt.plot(solution_matrix[:, 0], solution_matrix[:, 1] * 180 / np.pi)
plt.title("Theta vs Time")
plt.xlabel("time (sec)")
plt.ylabel("theta (deg)")
plt.legend(['theta:angle of bike\nwrt to vertical'])
plt.savefig("Theta-vs-Time.png")
plt.close()
# plot for theta_dot vs time
plt.figure(2)
plt.plot(solution_matrix[:, 0], solution_matrix[:, 2] * 180 / np.pi)
plt.title("Theta_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("theta_dot (deg/sec)")
plt.savefig("Theta_dot-vs-Time.png")
plt.close()
# plot for theta_dot_dot vs time
plt.figure(3)
plt.plot(solution_matrix[:, 0], del_theta_dot(solution_matrix) * 180 / np.pi)
plt.title("Theta_dot_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("theta_dot_dot (deg/sec^2)")
plt.savefig("Theta_dot_dot-vs-Time.png")
plt.close()
# plot for alpha vs time
plt.figure(4)
plt.plot(solution_matrix[:, 0], solution_matrix[:, 3] * 180 / np.pi)
plt.title("Alpha vs Time")
plt.xlabel("time (sec)")
plt.ylabel("alpha (deg)")
plt.legend(['alpha:angle of gyroscope\nnormal to\nbase of frame'])
plt.savefig("Alpha-vs-Time.png")
plt.close()
# plot for alpha_dot vs time
plt.figure(5)
plt.plot(solution_matrix[:, 0], del_alpha(solution_matrix) * 180 / np.pi)
plt.title("Alpha_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("alpha_dot (deg/sec)")
plt.savefig("Alpha_dot-vs-Time.png")
plt.close()
# plot for alpha_dot vs time
plt.figure(6)
plt.plot(solution_matrix[:, 0], del_alpha_dot(solution_matrix) * 180 / np.pi)
plt.title("Alpha_dot_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("alpha_dot_dot (deg/sec^2)")
plt.savefig("Alpha_dot_dot-vs-Time.png")
plt.close()
# plot for phi_dot vs time
plt.figure(7)
plt.plot(solution_matrix[:, 0], phi_dot(solution_matrix) * 30 / np.pi)
plt.title("Phi_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("phi_dot (deg/sec)")
plt.savefig("Phi_dot-vs-Time.png")
plt.close()
# plot for phi_dot_dot vs time
plt.figure(8)
plt.plot(solution_matrix[:, 0], del_phi_dot(solution_matrix) * 30 / np.pi)
plt.title("Phi_dot_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("phi_dot_dot (deg/sec^2)")
plt.savefig("Phi_dot_dot-vs-Time.png")
plt.close()

# plot for momentum_theta vs time
plt.figure(9)
plt.plot(solution_matrix[:, 0], momentum_theta(solution_matrix))
plt.title("momentum_theta vs Time")
plt.savefig("momentum_theta-vs-Time.png")
plt.close()
# plot for momentum_alpha vs time
plt.figure(10)
plt.plot(solution_matrix[:, 0], momentum_alpha(solution_matrix))
plt.title("momentum_alpha vs Time")
plt.savefig("momentum_alpha-vs-Time.png")
plt.close()

# plot for momentum_theta vs time
plt.figure(11)
plt.plot(solution_matrix[:, 1], momentum_theta(solution_matrix))
plt.title("momentum_theta vs theta")
plt.savefig("momentum_theta-vs-theta.png")
plt.close()

# plot for momentum_theta vs time
plt.figure(12)
plt.plot(solution_matrix[:, 3], momentum_theta(solution_matrix))
plt.title("momentum_theta vs alpha")
plt.savefig("momentum_theta-vs-alpha.png")
plt.close()

# plot for momentum_alpha vs time
plt.figure(13)
plt.plot(solution_matrix[:, 3], momentum_alpha(solution_matrix))
plt.title("momentum_alpha vs alpha")
plt.savefig("momentum_alpha-vs-alpha.png")
plt.close()

# plot for momentum_alpha vs time
plt.figure(14)
plt.plot(solution_matrix[:, 1], momentum_alpha(solution_matrix))
plt.title("momentum_alpha vs theta")
plt.savefig("momentum_alpha-vs-theta.png")
plt.close()

# plot for alpha vs theta
plt.figure(15)
plt.plot(solution_matrix[:, 1], solution_matrix[:, 3])
plt.title("alpha vs theta")
plt.savefig("alpha-vs-theta.png")
plt.close()
# plot for momentum_alpha vs time
plt.figure(16)
plt.plot(momentum_theta(solution_matrix), momentum_alpha(solution_matrix))
plt.title("momentum_alpha vs momentum_theta")
plt.savefig("momentum_alpha-vs-momentum_theta.png")
plt.close()

# plot for torque vs time
plt.figure(17)
plt.plot(solution_matrix[:, 0], torque(solution_matrix))
plt.title("Torque vs Time")
plt.xlabel("time (sec)")
plt.ylabel("torque (Nm)")
plt.legend(['torque:torque applied on gyroscope'])
plt.savefig("Torque-vs-Time.png")
plt.close()
# plot for power vs time
plt.figure(18)
plt.plot(solution_matrix[:, 0], torque_mat * alpha_dot_mat)
plt.title("Power vs Time")
plt.xlabel("time (sec)")
plt.ylabel("Power (Watt)")
plt.legend(['power:power supplied by the motor'])
plt.savefig("Power-vs-Time.png")
plt.close()

# plot for power vs time
plt.figure(19)
plt.plot(solution_matrix[:, 0], energy(solution_matrix))
plt.title("Energy vs Time")
plt.xlabel("time (sec)")
plt.ylabel("Energy (Joules)")
plt.savefig("Energy-vs-Time.png")
plt.close()

# plot for voltage vs time
plt.figure(20)
plt.plot(solution_matrix[:, 0], voltage)
plt.title("Voltage vs Time")
plt.xlabel("time (sec)")
plt.ylabel("voltage (V)")
plt.legend(['voltage:voltage applied\nacross motor circuit'])
plt.savefig("Voltage-vs-Time.png")
plt.close()

# plot for current vs time
plt.figure(21)
plt.plot(solution_matrix[:, 0], current)
plt.title("Current vs Time")
plt.xlabel("time (sec)")
plt.ylabel("current (Amp)")
plt.legend(['current:current in the motor'])
plt.savefig("Current-vs-Time.png")
plt.close()

plt.figure(22)
plt.plot(solution_matrix[:int(iter_num / 4), 0],
         solution_matrix[:int(iter_num / 4), 1] * 180 / np.pi)
plt.plot(solution_matrix[:int(iter_num / 4), 0],
         solution_matrix[:int(iter_num / 4), 2] * 180 / np.pi)
plt.plot(solution_matrix[:int(iter_num / 4), 0],
         solution_matrix[:int(iter_num / 4), 3] * 180 / np.pi)
plt.plot(solution_matrix[:int(iter_num / 4), 0],
         alpha_dot_mat[:int(iter_num / 4)] * 180 / np.pi)
plt.legend(['theta', 'theta dot', 'alpha', 'alpha dot'])
plt.xlabel("time (sec)")
plt.ylabel("angle (deg or deg/sec)")
plt.title("Combined Plot")
plt.savefig("Combined-Plot.png")
plt.close()
# plt.show()

I_holder = 28211.97 * 10**(-9)
h_holder = 0.1
force_on_gyro = (torque_mat - I_holder * alpha_dot_mat**2) / (h_holder)

plt.figure(23)
plt.plot(solution_matrix[:, 0], force_on_gyro)
plt.title("Force on Gyro End vs Time")
plt.xlabel("time (sec)")
plt.ylabel("Force (N)")
plt.legend(['force:force on the ends of gyro'])
plt.savefig("Force_on_gyro-vs-Time.png")
plt.close()
