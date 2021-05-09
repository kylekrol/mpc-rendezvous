from . import FixedHorizonController, VariableHorizonController, utilities

from psim import Configuration, Simulation
from psim.sims import OrbitMpcRendezvous

import numpy as np
import scipy as sp

import lin


# Number of steps taken between manuevers
T = 1765

# Prediction horizon
N = 100

# Satellite mass
m = 3.6

# Base configuration for simulations
configs = ['sensors/base', 'truth/base', 'truth/near_field']
configs = ['lib/psim/config/parameters/' + config + '.txt' for config in configs]
config = Configuration(configs)

config['truth.follower.m'] = m
config['truth.dt.ns'] = 100000000

# Fields being logged to CSV from the simulation
fields = [
    'truth.t.ns',
    'truth.follower.hill.dr', 'truth.follower.hill.dv',
    'fc.follower.relative_orbit.is_valid',
    'fc.follower.relative_orbit.r.hill', 'fc.follower.relative_orbit.v.hill'
]

# Output CSV file
filename = 'logs/logs.csv'

# Costs
Q = sp.sparse.diags([1e4, 1e4, 1e4, 1e4, 1e4, 1e4])
R = sp.sparse.diags([100.0, 100.0, 100.0])
rho = 10.0
J = 1.0e-16

# Maximum horizon
Nmax = 100

# Maximum allowable impulse
umax = 0.025

# Initialize logging arrays
logs = {
    'fc.follower.orbit.J.hill': list(),
    'fc.follower.orbit.J.hill.norm': list(),
    'fc.follower.orbit.horizon': list()
}
for field in fields:
    logs[field] = list()

# Construct the simulation
config['sensors.follower.cdgps.model_range'] = False
sim = Simulation(OrbitMpcRendezvous, config)
dt = float(sim['truth.dt.ns'] * T) / 1e9

# Construct the solver
controller = VariableHorizonController(Nmax, J, Q, R, rho, umax)

sim.step()
sim.step()

# Main simulation loop
for _ in range(250):

    # Mean motion of the follower
    r = sim['truth.follower.orbit.r.eci']
    v = sim['truth.follower.orbit.v.eci']
    n = lin.norm(lin.cross(r, v) / lin.fro(r))

    # Run the controller
    dr = sim['fc.follower.relative_orbit.r.hill']
    dv = sim['fc.follower.relative_orbit.v.hill']

    if not sim['sensors.follower.cdgps.valid']:
        print(f'Lost CDGPS; ||dr|| = {lin.norm(dr)}')

    if lin.norm(dr) < 0.40:
        print('Rendezvous complete')
        break

    A = utilities.cw(dt, n)
    B = -A[:, 3:6] / m

    controller.run(A, B, dr, dv)
    u = lin.Vector3(controller.control)
    sim['fc.follower.orbit.J.hill'] = u

    logs['fc.follower.orbit.J.hill'].append(u)
    logs['fc.follower.orbit.J.hill.norm'].append(lin.norm(u))
    logs['fc.follower.orbit.horizon'].append(controller.horizon)
    for field in fields:
        logs[field].append(sim[field])

    sim.step()

    for _ in range(T-1):
        logs['fc.follower.orbit.J.hill'].append(lin.Vector3())
        logs['fc.follower.orbit.J.hill.norm'].append(0)
        logs['fc.follower.orbit.horizon'].append(0)
        for field in fields:
            logs[field].append(sim[field])

        sim.step()

# Save data to a CSV
with open(filename, 'w') as csv:
    keys = sorted(logs.keys())
    suffixes = ['x', 'y', 'z', 'w']

    for key in keys:
        value = logs[key][0]
        if type(value) in [lin.Vector2, lin.Vector3, lin.Vector4]:
            for j in range(value.size()):
                csv.write(f'{key}.{suffixes[j]},')
        else:
            csv.write(f'{key},')
    csv.write('\n')

    for i in range(len(logs['truth.t.ns'])):
        for key in keys:
            value = logs[key][i]
            if type(value) in [lin.Vector2, lin.Vector3, lin.Vector4]:
                for j in range(value.size()):
                    csv.write(f'{value[j]},')
            else:
                csv.write(f'{value},')
        csv.write('\n')
