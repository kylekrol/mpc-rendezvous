from .utilities import cw

from psim import Configuration, Simulation
from psim.sims import OrbitMpcRendezvous

import lin


class Simulation(object):

    def __init__(self, m, dt, T, controller):
        super(Simulation, self).__init__()

        configs = ['sensors/base', 'truth/base', 'truth/near_field']
        configs = ['lib/psim/config/parameters/' + config + '.txt' for config in configs]
        config = Configuration(configs)

        config['truth.dt.ns'] = dt
        config['truth.follower.m'] = m

        self.controller_interval = T / dt
        self.controller = controller

    @property
    def dr(self):
        raise NotImplementedError()

    @property
    def dv(self):
        raise NotImplementedError()

    def step(self):
        pass
