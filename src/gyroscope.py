#!/usr/bin/env python
import matplotlib.pyplot as plt
import os
import differential_equations
import numpy as np

#Gyroscope parameters
rod_rad = 1*10**-2
disk_rad = 11*10**-2
rod_len = 10*10**-2
disk_len = 4*10**-2
disk_pos = 6.5*10**-2
density = 7.85*10**3

#mass of gyroscope calculations
gyro_mass = density*np.pi \
    * (rod_rad**2*rod_len+(disk_rad**2-rod_rad**2)*disk_len)

#center of mass of gyroscope calculations
x_mass_centre = (density*np.pi/gyro_mass)*(rod_rad**2*rod_len**2/2
                                           + (disk_rad**2-rod_rad**2)
                                           * disk_len*(disk_pos+disk_len/2))

#Moment of inertia of gyroscope about endpoint of shaft
I_endpoint = density*np.pi*(rod_rad**4*rod_len/4+((disk_rad**4-rod_rad**4)*disk_len/4)
                            + (rod_rad**2 * rod_len**3)
                            / 3+(disk_rad**2-rod_rad**2)
                            * (disk_len**3+3*disk_pos**2*disk_len
                               + 3*disk_len**2*disk_pos)/3)

#Moment of inertia of gyroscope about center of mass using parallel axis theorem
#This moment of inertia is about both alpha and zeta axis.
I_xi_xi = I_endpoint - gyro_mass*x_mass_centre**2

#This moment of inertia is about both phi axis.
I_phi_phi = (1/2)*density*np.pi*(disk_len*disk_rad**4
                                 - (rod_len-disk_len)*rod_rad**4)

# bike parameters
#mass of bike(100kg) plus addition payload such as weight of people and other things.
bike_mass = 250
grav_accel = 9.81
#center of mass of bike plus payload just choosen to be at saddle of bike as
#center of mass will be around that position
#Just a rough approximation
bike_mass_center = 0.3

#The position where gyroscope is pivoted. This position is assigned relative to
#the ground.
gyro_pos = 0.45
#position where pivot is located on the shaft relative to endpoint of shaft of
#gyroscope
pivot_pos = 1*10**-2
#length from pivot to gyroscope center of mass
pivot_len = x_mass_centre-pivot_pos
#angular velocity of gyroscope about phi axis in radian/s
gyro_vel = 700
#number of iterations
iter_num = 30000
#moment of inertia of bike plus that of payload
#estimated as modeling people as bars
#Calculated bike moment of inertia using cading software and finding
#approximate radius of gyration.
I_bike = 45

# intial conditions
#difference between time interval of individual iterations
time_diff = 0.001
#Starting time
init_time = 0
# intial condition matrix 1st entry is that of independent variable latter
#corresponds to dependent variables
init_cond = np.array([init_time, 0.1, 0.1, 0.1, 0])
#conjugate momentum in phi
momentum_phi = I_phi_phi * (gyro_vel+init_cond[2]*np.sin(init_cond[3]))

# functions corresponding to differential equations


def del_theta(x_input):
    """This function corresponds to derivative of theta."""
    theta_dot = x_input[2]
    return theta_dot


def del_theta_dot(x_input):
    """This function corresponds to second derivative of theta."""
    # This if elif statement is used as to allow both one and 
    # two dimension array input, as two dimension array computation
    # is useful in graphs
    if x_input.ndim == 1:
        theta = x_input[1]
        theta_dot = x_input[2]
        alpha = x_input[3]
        alpha_dot = x_input[4]
    elif x_input.ndim == 2:
        theta = x_input[:, 1]
        theta_dot = x_input[:, 2]
        alpha = x_input[:, 3]
        alpha_dot = x_input[:, 4]

    bike_weight_torq = bike_mass*grav_accel*bike_mass_center*np.sin(theta)
    gyro_weight_torq = 2*gyro_mass*grav_accel \
        * (gyro_pos+pivot_len*np.cos(alpha))*np.sin(theta)

    restoring_torq = 2*momentum_phi*np.cos(alpha)*alpha_dot
    other_torq = 4*(I_xi_xi*np.cos(alpha)+gyro_mass
                    * (gyro_pos+pivot_len*np.cos(alpha))*pivot_len) \
        * theta_dot*alpha_dot*np.sin(alpha)

    effec_inertia = I_bike+2*gyro_mass*(gyro_pos+pivot_len*np.cos(alpha))**2 \
        + 2*I_xi_xi*(np.cos(alpha))**2
    theta_dot_dot = (bike_weight_torq + gyro_weight_torq - restoring_torq
                     + other_torq)/effec_inertia

    return theta_dot_dot


def del_alpha(x_input):
    """This function corresponds to derivative of alpha."""
    alpha_dot = x_input[4]
    return alpha_dot


def del_alpha_dot(x_input):
    """This function corresponds to second derivative of alpha."""
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

    mass_torq = gyro_mass*pivot_len*(gyro_pos+pivot_len*np.cos(alpha)) \
        * np.sin(alpha)*theta_dot**2

    restoring_torq = momentum_phi*theta_dot*np.cos(alpha)
    inertia_torq = I_xi_xi*theta_dot**2*np.cos(alpha)*np.sin(alpha)
    weight_torq = gyro_mass*grav_accel*pivot_len*np.sin(alpha) \
        * np.cos(theta)

    effec_inertia = gyro_mass*pivot_len**2+I_xi_xi
    alpha_dot_dot = (-mass_torq+restoring_torq-inertia_torq+weight_torq) \
        / effec_inertia

    return alpha_dot_dot


# solution of differential equations
eqn = differential_equations.DifferentialEquations(del_theta, del_theta_dot,
                                                   del_alpha, del_alpha_dot)

#solution matrix
solution_matrix = eqn.solution_differential_eqn(time_diff, iter_num, init_cond)

theta_dot_dot_mat = del_theta_dot(solution_matrix)
momentum_phi_mat = momentum_phi*np.ones(len(solution_matrix[:, 0]))


def del_phi_dot(x_input):
    """This function corresponds to second derivative of phi."""
    theta_dot = x_input[:, 2]
    theta_dot_dot = theta_dot_dot_mat
    alpha = x_input[:, 3]
    alpha_dot = x_input[:, 4]

    phi_dot_dot = -theta_dot_dot \
        * np.sin(alpha)-theta_dot*alpha_dot*np.cos(alpha)
    return phi_dot_dot


def energy(x_input):
    """This is energy function of the system."""
    theta = x_input[:, 1]
    theta_dot = x_input[:, 2]
    alpha = x_input[:, 3]
    alpha_dot = x_input[:, 4]
    momentum_phi = momentum_phi_mat

    energy_mat = (1/2)*I_bike*theta_dot**2 \
        + gyro_mass*(((gyro_pos+pivot_len*np.cos(alpha))**2)*theta_dot**2) \
        + I_xi_xi*(np.cos(alpha)*theta_dot)**2 \
        + (gyro_mass*pivot_len**2+I_xi_xi)*(alpha_dot**2) \
        + (momentum_phi**2/I_phi_phi) \
        + bike_mass*grav_accel*bike_mass_center*np.cos(theta) \
        + 2*gyro_mass*grav_accel*(gyro_pos+pivot_len
                                  * np.cos(alpha))*np.cos(theta)
    return energy_mat


def phi_dot(x_input):
    """This function corresponds to derivative of phi."""
    theta_dot = x_input[:, 2]
    alpha = x_input[:, 3]
    momentum_phi = momentum_phi_mat
    phi_dot_mat = (momentum_phi/I_phi_phi) - theta_dot*np.sin(alpha)
    return phi_dot_mat


def momentum_theta(x_input):
    """This function corresponds to conjugate momentum for theta."""
    theta_dot = x_input[:, 2]
    alpha = x_input[:, 3]

    momentum_theta_mat = I_bike*theta_dot \
        + 2*gyro_mass*((gyro_pos+pivot_len*np.cos(alpha))**2)*theta_dot \
        + 2*momentum_phi*np.sin(alpha)+2*I_xi_xi \
        * ((np.cos(alpha))**2)*theta_dot

    return momentum_theta_mat


def momentum_alpha(x_input):
    """This function corresponds to conjugate momentum for alpha."""
    alpha_dot = x_input[:, 4]
    return (I_xi_xi+gyro_mass*pivot_len**2) * (alpha_dot)


try:
    os.mkdir("gyroscope_graphs")
except:
    pass
try:
    os.chdir('gyroscope_graphs')
except:
    exit()

plt.figure(1)
plt.plot(solution_matrix[:int(iter_num/4), 0],
         solution_matrix[:int(iter_num/4), 1]*180/np.pi)
plt.plot(solution_matrix[:int(iter_num/4), 0],
         solution_matrix[:int(iter_num/4), 2]*180/np.pi)
plt.plot(solution_matrix[:int(iter_num/4), 0],
         solution_matrix[:int(iter_num/4), 3]*180/np.pi)
plt.plot(solution_matrix[:int(iter_num/4), 0],
         solution_matrix[:int(iter_num/4), 4]*180/np.pi)
plt.legend(['theta', 'theta dot', 'alpha', 'alpha dot'])
plt.title("Combined Plot")
plt.xlabel("time (sec)")
plt.ylabel("angle (deg or deg/sec)")
plt.savefig("Combined-Plot.png")
plt.close()

#plot for theta vs time
plt.figure(2)
plt.plot(solution_matrix[:, 0], solution_matrix[:, 1]*180/np.pi)
plt.title("Theta vs Time")
plt.xlabel("time (sec)")
plt.ylabel("theta (deg)")
plt.savefig("Theta-vs-Time.png")
plt.close()
#plot for theta_dot vs time
plt.figure(3)
plt.plot(solution_matrix[:, 0], solution_matrix[:, 2]*180/np.pi)
plt.title("Theta_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("theta_dot (deg/sec)")
plt.savefig("Theta_dot-vs-Time.png")
plt.close()
#plot for theta_dot_dot vs time
plt.figure(4)
plt.plot(solution_matrix[:, 0], del_theta_dot(solution_matrix)*180/np.pi)
plt.title("Theta_dot_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("theta_dot_dot (deg/sec^2)")
plt.savefig("Theta_dot_dot-vs-Time.png")
plt.close()
#plot for alpha vs time
plt.figure(5)
plt.plot(solution_matrix[:, 0], solution_matrix[:, 3]*180/np.pi)
plt.title("Alpha vs Time")
plt.xlabel("time (sec)")
plt.ylabel("alpha (deg)")
plt.savefig("Alpha-vs-Time.png")
plt.close()
#plot for alpha_dot vs time
plt.figure(6)
plt.plot(solution_matrix[:, 0], solution_matrix[:, 4]*180/np.pi)
plt.title("Alpha_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("alpha_dot (deg/sec)")
plt.savefig("Alpha_dot-vs-Time.png")
plt.close()
#plot for alpha_dot vs time
plt.figure(7)
plt.plot(solution_matrix[:, 0], del_alpha_dot(solution_matrix)*180/np.pi)
plt.title("Alpha_dot_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("alpha_dot_dot (deg/sec^2)")
plt.savefig("Alpha_dot_dot-vs-Time.png")
plt.close()
#plot for phi_dot vs time
plt.figure(8)
plt.plot(solution_matrix[:, 0], phi_dot(solution_matrix)*30/np.pi)
plt.title("Phi_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("phi_dot (deg/sec)")
plt.savefig("Phi_dot-vs-Time.png")
plt.close()
#plot for phi_dot_dot vs time
plt.figure(9)
plt.plot(solution_matrix[:, 0], del_phi_dot(solution_matrix)*30/np.pi)
plt.title("Phi_dot_dot vs Time")
plt.xlabel("time (sec)")
plt.ylabel("phi_dot_dot (deg/sec^2)")
plt.savefig("Phi_dot_dot-vs-Time.png")
plt.close()


#plot for momentum_theta vs time
plt.figure(10)
plt.plot(solution_matrix[:, 0], momentum_theta(solution_matrix))
plt.title("momentum_theta vs Time")
plt.savefig("momentum_theta-vs-Time.png")
plt.close()
#plot for momentum_alpha vs time
plt.figure(11)
plt.plot(solution_matrix[:, 0], momentum_alpha(solution_matrix))
plt.title("momentum_alpha vs Time")
plt.savefig("momentum_alpha-vs-Time.png")
plt.close()


#plot for momentum_theta vs time
plt.figure(12)
plt.plot(solution_matrix[:, 1], momentum_theta(solution_matrix))
plt.title("momentum_theta vs theta")
plt.savefig("momentum_theta-vs-theta.png")
plt.close()

#plot for momentum_theta vs time
plt.figure(13)
plt.plot(solution_matrix[:, 3], momentum_theta(solution_matrix))
plt.title("momentum_theta vs alpha")
plt.savefig("momentum_theta-vs-alpha.png")
plt.close()

#plot for momentum_alpha vs time
plt.figure(14)
plt.plot(solution_matrix[:, 3], momentum_alpha(solution_matrix))
plt.title("momentum_alpha vs alpha")
plt.savefig("momentum_alpha-vs-alpha.png")
plt.close()

#plot for momentum_alpha vs time
plt.figure(15)
plt.plot(solution_matrix[:, 1], momentum_alpha(solution_matrix))
plt.title("momentum_alpha vs theta")
plt.savefig("momentum_alpha-vs-theta.png")
plt.close()


#plot for alpha vs theta
plt.figure(16)
plt.plot(solution_matrix[:, 1], solution_matrix[:, 3])
plt.title("alpha vs theta")
plt.savefig("alpha-vs-theta.png")
plt.close()
#plot for momentum_alpha vs time
plt.figure(17)
plt.plot(momentum_theta(solution_matrix), momentum_alpha(solution_matrix))
plt.title("momentum_alpha vs momentum_theta")
plt.savefig("momentum_alpha-vs-momentum_theta.png")
plt.close()


#plot for power vs time
plt.figure(18)
plt.plot(solution_matrix[:, 0], energy(solution_matrix))
plt.title("Energy vs Time")
plt.xlabel("time (sec)")
plt.ylabel("Energy (Joules)")
plt.savefig("Energy-vs-Time.png")
plt.close()
