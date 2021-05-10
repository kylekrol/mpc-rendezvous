from mpc import ClohessyWiltshireDynamics, LPController, QCQPController, VariableHorizonController
from mpc.utilities import cw

import numpy as np
import scipy as sp

import datetime
import math
import os
import sys


### Configuration Parameters ###################################################

# Mode -- either 'simulate' or 'plan'
mode = "plan"

# Simulation timestep (0.1 s), satellite mass, and target mean motion.
dt = 100000000
m = 3.6
n = 2.0 * math.pi / (90.0 * 60.0)

# Process noise, measurement noise, and initial state
sw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
sv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
x0 = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Controller frequency, horizon limit, constant stage cost, final stage cost,
# quadratic control cost, L1 control cost, and largest allowable impulse
T = 5 * 60 * 10
N = 20
J = 0.0
rho = 0.0
umax = 0.025

# Function to generate CW dynamics
def make_cw():
    return ClohessyWiltshireDynamics(dt, n, m, sw, sv, x0)

# Function to generate a QCQP controller
def make_qcqp():
    # QCQP specific configuration
    Q = sp.sparse.diags([100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0])
    R = sp.sparse.diags([1.0, 1.0, 1.0])

    #return QCQPController(N, J, Q, R, rho, umax)
    return VariableHorizonController(N, QCQPController, J, Q, R, rho, umax)

# Function to generate an LP controller
def make_lp():
    # LP specific configuration
    rtol = 0.01
    vtol = 0.001

    return VariableHorizonController(N, LPController, J, rho, rtol, vtol, umax)

# Stopping condition
def stop(dynamics):
    return np.norm(dynamics.true_dr) <= 0.3 and np.norm(dynamics.true_dv) <= 0.003

# Dynamics and controller selection
make_dynamics = make_cw
make_controller = make_qcqp

# Logging file directory
logging_directory = "logs/cw-qcqp-{}".format(datetime.datetime.now().strftime("%m%d%Y-%H%M%S"))

################################################################################

# Generate our dynamics and controller
dynamics = ClohessyWiltshireDynamics(dt, n, m, sw, sv, x0)
controller = make_controller()

# Handle if we're only looking to plan here
if mode == 'plan':
    import matplotlib.pyplot as plt

    A = cw((T * dt) / 1.0e9, n)
    B = A[:, 3:6] / m

    controller.run(A, B, x0, verbose=False)

    print("Results:")
    print("\tControl    = {}".format(controller.control))
    print("\tTotal cost = {}".format(controller.total_cost))
    print("\tStatus     = {}".format(controller.status))
    print("\tHorizon    = {}".format(controller.horizon))

    X, U = controller.trajectory

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X[0, :], X[1, :], X[2, :])
    ax.set_xlabel('dr.x')
    ax.set_ylabel('dr.y')
    ax.set_zlabel('dr.z')
    fig.show()

    fig = plt.figure()
    plt.plot(X[0, :], label='dr.x')
    plt.plot(X[1, :], label='dr.y')
    plt.plot(X[2, :], label='dr.z')
    plt.legend()
    plt.xlabel('t')
    fig.show()

    fig = plt.figure()
    plt.plot(X[3, :], label='dv.x')
    plt.plot(X[4, :], label='dv.y')
    plt.plot(X[5, :], label='dv.z')
    plt.legend()
    plt.xlabel('t')
    fig.show()

    fig = plt.figure()
    plt.plot(U[0, :], label='J.x')
    plt.plot(U[1, :], label='J.y')
    plt.plot(U[2, :], label='J.z')
    plt.legend()
    plt.xlabel('t')
    fig.show()

    C = controller.stage_costs

    fig = plt.figure()
    plt.plot(C)
    plt.xlabel('t')
    plt.ylabel('Cost')
    plt.show()

    sys.exit(0)


# Dynamics logs
dynamics_logs = [
    "mean_motion", "measured_dr", "measured_dv", "time_s", "true_dr", "true_dv"
]
dynamics_logs = {name: list() for name in dynamics_logs}

# Controller logs
dynamics_logs = [
    "mean_motion", "measured_dr", "measured_dv", "time_s", "true_dr", "true_dv"
]
dynamics_logs = {name: list() for name in dynamics_logs}

# Generate controller dynamics
A = cw((T * dt) / 1.0e9, n)
B = A[:, 3:6] / m
