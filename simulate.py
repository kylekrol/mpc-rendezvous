from mpc import ClohessyWiltshireDynamics, LPController, QCQPController, PSimDynamics, VariableHorizonController
from mpc.utilities import cw

import numpy as np
import scipy as sp

import datetime
import math
import os


### Configuration Parameters ###################################################

# Mode -- either 'simulate', 'plan', or 'monte-carlo'
mode = 'simulate'

# Simulation timestep (0.1 s), satellite mass, and target mean motion.
dt = 100000000
m = 3.6

# Initial state
x0 = np.array([25.0, 500.0, 10.0, 0.0, 0.015, 0.0])

# Controller frequency, horizon limit, constant stage cost, final stage cost,
# quadratic control cost, L1 control cost, and largest allowable impulse
T = 5 * 60 * 10
N = 72
umax = 0.025

# Function to generate CW dynamics
def make_cw(x0):
    n = 2.0 * math.pi / (90.0 * 60.0)
    sw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    sv = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    return ClohessyWiltshireDynamics(dt, n, m, sw, sv, x0)

# Function to generate PSim dynamics
def make_psim(x0):
    return PSimDynamics(dt, m, x0)

# Function to generate a QCQP controller
def make_qcqp():
    ## Solely L2 norm control cost
    # Q = sp.sparse.diags([100.0, 100.0, 100.0, 10000.0, 10000.0, 10000.0])
    # R = sp.sparse.diags([1.0, 1.0, 1.0])
    # rho = 0.0
    # J = 1.0e-4
    ##

    ## Joint L2 and L1 norm control cost
    # Q = sp.sparse.diags([100.0, 100.0, 100.0, 10000.0, 10000.0, 10000.0])
    # R = sp.sparse.diags([0.25, 0.25, 0.25])
    # rho = 3.0 * 0.025 / 4.0
    # J = 0.4e-4
    ##

    ## Solely L1 norm control cost
    Q = sp.sparse.diags([100.0, 100.0, 100.0, 10000.0, 10000.0, 10000.0])
    R = sp.sparse.diags([0.0, 0.0, 0.0])
    rho = 0.025
    J = 0.9e-4
    ##

    return VariableHorizonController(N, QCQPController, J, Q, R, rho, umax)

# Function to generate an LP controller
def make_lp():
    # LP specific configuration
    rtol = 0.01
    vtol = 0.001
    rho = 0.025
    J = 1.0e-4

    return VariableHorizonController(N, LPController, J, rho, rtol, vtol, umax)

# Stopping condition
def stop(dynamics):
    return np.linalg.norm(dynamics.true_dr) <= 0.4 and np.linalg.norm(dynamics.true_dv) <= 0.004

# Dynamics and controller selection
make_dynamics = make_psim
make_controller = make_qcqp

# Logging file directory
logging_directory = 'logs/measured-l1-{}'.format(datetime.datetime.now().strftime('%m%d%Y-%H%M%S'))

################################################################################

# Generate our dynamics and controller
dynamics = make_dynamics(x0)
controller = make_controller()

if mode == 'plan':
    print('Starting planning!')

    import matplotlib.pyplot as plt

    A = cw((T * dt) / 1.0e9, dynamics.mean_motion)
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

elif mode == 'simulate':
    print('Starting simulation!')

    # Dynamics logs
    dynamics_logs = [
        "mean_motion", "time_s", "true_dr", "true_dv"
    ]
    dynamics_logs = {name: list() for name in dynamics_logs}

    # Controller logs
    controller_logs = [
        "control", "total_cost", "horizon", "time_s"
    ]
    controller_logs = {name: list() for name in controller_logs}

    steps = 0
    while not stop(dynamics) and steps < 10000000:
        # Drift stage
        for i in range(T - 1):
            dynamics.step()
            if i % 10 == 0:
                for key, values in dynamics_logs.items():
                    values.append(getattr(dynamics, key))

            steps = steps + 1

            if stop(dynamics):
                break

        if stop(dynamics):
            break

        # Control stage
        A = cw((T * dt) / 1.0e9, dynamics.mean_motion)
        B = A[:, 3:6] / m
        r0 = dynamics.measured_dr
        v0 = dynamics.measured_dv
        x0 = np.array([r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]])

        controller.run(A, B, x0)
        u = -controller.control if isinstance(dynamics, PSimDynamics) else controller.control
        dynamics.step(u)
        for key, values in controller_logs.items():
            if key != 'time_s' and key != 'control' and key != 'computation':
                values.append(getattr(controller, key))
        controller_logs['control'].append(u)
        controller_logs['time_s'].append(dynamics.time_s)
        for key, values in dynamics_logs.items():
            values.append(getattr(dynamics, key))

        steps = steps + 1
        print('Control Horizon={}, Steps={}'.format(controller.horizon, steps))

    os.makedirs(logging_directory)
    def logging(filename, logs):
        with open(logging_directory + '/' + filename, 'w') as csv:
            keys = sorted(logs.keys())
            suffixes = ['x', 'y', 'z', 'w']

            for key in keys:
                value = logs[key][0]
                if type(value) is np.ndarray:
                    for j in range(value.size):
                        csv.write(f'{key}.{suffixes[j]},')
                else:
                    csv.write(f'{key},')
            csv.write('\n')

            for i in range(len(logs['time_s'])):
                for key in keys:
                    value = logs[key][i]
                    if type(value) is np.ndarray:
                        for j in range(value.size):
                            csv.write(f'{value[j]},')
                    else:
                        csv.write(f'{value},')
                csv.write('\n')

    logging('controller.csv', controller_logs)
    logging('dynamics.csv', dynamics_logs)

elif mode == 'monte-carlo':
    print('Starting Monte Carlo!')

    samples = 100
    controller = make_controller()

    dvs = list()
    stages = list()
    for _ in range(samples):

        r = 20.0 * np.random.randn()
        v = (-500.0 if np.random.rand() > 0.5 else 500.0) + 20.0 * np.random.randn()
        n = 10.0 * np.random.randn()
        x0 = np.array([r, v, n, 0.0, 0.0, 0.0])

        dynamics = make_dynamics(x0)

        dv = 0.0
        stage = 0
        while not stop(dynamics) and stage < 1000:

            for i in range(T - 1):
                dynamics.step()
                if stop(dynamics):
                    break
            if stop(dynamics):
                break

            # Control stage
            A = cw((T * dt) / 1.0e9, dynamics.mean_motion)
            B = A[:, 3:6] / m
            r0 = dynamics.measured_dr
            v0 = dynamics.measured_dv
            x0 = np.array([r0[0], r0[1], r0[2], v0[0], v0[1], v0[2]])
            controller.run(A, B, x0)
            u = -controller.control if isinstance(dynamics, PSimDynamics) else controller.control
            dynamics.step(u)

            dv += np.linalg.norm(u)
            stage += 1

        if stage >= 1000:
            print('WARNING: Stage timed out!')
        print('Completed run with dv={} and stage={}'.format(dv, stage))

        dvs.append(dv)
        stages.append(stage)

    print('dv mean={} and std={}'.format(np.average(dvs), np.std(dvs)))
    print('stages mean={} and std={}'.format(np.average(stages), np.std(stages)))
