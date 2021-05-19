import matplotlib.pyplot as plt

import numpy as np
import pandas

import sys


if len(sys.argv) != 2:
    print('USAGE: python plots.py LOGS', sys.stderr)
    sys.exit(-1)

directory = sys.argv[1]
controller = pandas.read_csv(directory + '/controller.csv')
dynamics = pandas.read_csv(directory + '/dynamics.csv')


fig = plt.figure()

ax = fig.add_subplot(221)
ax.bar(controller['time_s'], controller['control.x'], width=300.0)
ax.set_xlabel('$t$ (s)')
ax.set_ylabel('$u_x$ (Ns)')
ax.set_ylim(-0.025, 0.025)
ax.grid('major')
ax.set_axisbelow(True)

ax = fig.add_subplot(222)
ax.bar(controller['time_s'], controller['control.y'], width=300.0)
ax.set_xlabel('$t$ (s)')
ax.set_ylabel('$u_y$ (Ns)')
ax.set_ylim(-0.025, 0.025)
ax.grid('major')
ax.set_axisbelow(True)

ax = fig.add_subplot(223)
ax.bar(controller['time_s'], controller['control.z'], width=300.0)
ax.set_xlabel('$t$ (s)')
ax.set_ylabel('$u_z$ (Ns)')
ax.set_ylim(-0.025, 0.025)
ax.grid('major')
ax.set_axisbelow(True)

ax = fig.add_subplot(224)
ax.plot(controller['time_s'], controller['horizon'])
ax.set_xlabel('$t$ (s)')
ax.set_ylabel('Desired Horizon')
ax.set_ylim(0, 72)
ax.grid('major')
ax.set_axisbelow(True)

fig.tight_layout()
fig.show()


fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.plot(dynamics['true_dr.x'], dynamics['true_dr.y'], dynamics['true_dr.z'])
ax.set_xlabel('$\delta r_x$ (m)')
ax.set_ylabel('$\delta r_y$ (m)')
ax.set_zlabel('$\delta r_z$ (m)')

fig.tight_layout()
fig.show()


fig = plt.figure()

ax = fig.add_subplot(211)
ax.plot(dynamics['time_s'], dynamics['true_dr.x'], label='$\delta r_x$')
ax.plot(dynamics['time_s'], dynamics['true_dr.y'], label='$\delta r_y$')
ax.plot(dynamics['time_s'], dynamics['true_dr.z'], label='$\delta r_z$')
ax.legend()
ax.set_ylabel('Relative Position (m)')
ax.grid('major')
ax.set_axisbelow(True)

ax = fig.add_subplot(212)
ax.plot(dynamics['time_s'], dynamics['true_dv.x'], label='$\delta v_x$')
ax.plot(dynamics['time_s'], dynamics['true_dv.y'], label='$\delta v_y$')
ax.plot(dynamics['time_s'], dynamics['true_dv.z'], label='$\delta v_z$')
ax.legend()
ax.set_xlabel('$t$ (s)')
ax.set_ylabel('Relative Velocity (m/s)')
ax.grid('major')
ax.set_axisbelow(True)

fig.tight_layout()
plt.show()
