from .base import Dynamics

from psim import Configuration, Simulation
from psim.sims import OrbitMpcRendezvous

import lin
import numpy as np


def _leader_ics(dr_hill, dv_hill, w_earth_ecef, r_ecef, v_ecef):
    # Inertially stuck ECEF
    r_ecef0 = r_ecef
    v_ecef0 = v_ecef + lin.cross(w_earth_ecef, r_ecef0)

    # Angular rate of the will frame in the interially stuck ECEF frame
    w_hill_ecef0 = lin.cross(r_ecef0, v_ecef0) / lin.fro(r_ecef0)

    # Unit vectors
    r_ecef0_hat = r_ecef0 / lin.norm(r_ecef0)
    v_ecef0_hat = v_ecef0 - v_ecef0 * lin.dot(v_ecef0, r_ecef0_hat)
    v_ecef0_hat = v_ecef0_hat / lin.norm(v_ecef0_hat)
    n_ecef0_hat = lin.cross(r_ecef0_hat, v_ecef0_hat)

    # DCM conversion between frames
    Q_hill_ecef0 = lin.Matrix3x3([
        r_ecef0_hat[0], r_ecef0_hat[1], r_ecef0_hat[2],
        v_ecef0_hat[0], v_ecef0_hat[1], v_ecef0_hat[2],
        n_ecef0_hat[0], n_ecef0_hat[1], n_ecef0_hat[2],
    ])
    Q_ecef0_hill = lin.transpose(Q_hill_ecef0)

    dr_ecef = Q_ecef0_hill * dr_hill
    dv_ecef = Q_ecef0_hill * dv_hill - lin.cross(w_earth_ecef - w_hill_ecef0, dr_ecef)

    return r_ecef + dr_ecef, v_ecef + dv_ecef


class PSimDynamics(Dynamics):
    """Dynamics model implemented with PSim as a backend.
    """
    def __init__(self, dt, m, x0):
        super(PSimDynamics, self).__init__(dt)

        configs = ['sensors/base', 'truth/base', 'truth/near_field']
        configs = ['lib/psim/config/parameters/' + config + '.txt' for config in configs]
        config = Configuration(configs)

        config['truth.dt.ns'] = dt
        config['truth.follower.mass'] = m
        config['sensors.follower.cdgps.model_range'] = False
        config['fc.follower.relative_orbit.r.hill.alpha'] = 0.6
        config['fc.follower.relative_orbit.v.hill.alpha'] = 0.6

        sim = Simulation(OrbitMpcRendezvous, config)

        r_ecef, v_ecef = _leader_ics(
            lin.Vector3(x0[0:3]), lin.Vector3(x0[3:6]),
            sim['truth.earth.w'], sim['truth.follower.orbit.r.ecef'],
            sim['truth.follower.orbit.v.ecef']
        )

        config['truth.leader.orbit.r'] = r_ecef
        config['truth.leader.orbit.v'] = v_ecef

        self.sim = Simulation(OrbitMpcRendezvous, config)
        self.t0 = self.sim['truth.t.ns']

    @property
    def mean_motion(self):
        return lin.norm(self.sim['truth.follower.hill.w.eci'])

    @property
    def measured_dr(self):
        """Current, measured position of the follower spacecraft in the HILL
        frame.
        """
        if not self.sim['fc.follower.relative_orbit.is_valid']:
            raise RuntimeError('Invalid relative orbit estimate')

        return np.array(self.sim['fc.follower.relative_orbit.r.hill'])

    @property
    def measured_dv(self):
        """Current, measured velocity of the follower spacecraft in the HILL
        frame.
        """
        if not self.sim['fc.follower.relative_orbit.is_valid']:
            raise RuntimeError('Invalid relative orbit estimate')

        return np.array(self.sim['fc.follower.relative_orbit.v.hill'])

    @property
    def time_ns(self):
        """Current simulation time in nanoseconds.
        """
        return self.sim['truth.t.ns'] - self.t0

    @property
    def time_s(self):
        """Current simulation time in seconds.
        """
        return self.time_ns / 1.0e9

    @property
    def true_dr(self):
        """Current position of the follower spacecraft in the HILL frame.
        """
        return np.array(self.sim['truth.follower.hill.dr'])

    @property
    def true_dv(self):
        """Current velocity of the follower spacecraft in the HILL frame.
        """
        return np.array(self.sim['truth.follower.hill.dv'])

    def step(self, u=None):
        super(PSimDynamics, self).step()

        if u is not None:
            self.sim['fc.follower.orbit.J.hill'] = lin.Vector3(u)
        self.sim.step()
