from .base import Dynamics
from ..utilities import cw

import numpy as np


class ClohessyWiltshireDynamics(Dynamics):
    """Linear dynamics model implemented with the Clohessy-Wiltshire equations.
    """
    def __init__(self, dt, n, m, sw, sv, x0):
        super(ClohessyWiltshireDynamics, self).__init__(dt)

        # Process nose and sensor noise one sigma
        self.sw = sw
        self.sv = sv

        # Mean motion
        self.n = n

        # Dynamics, state, and measurement
        self.A = cw(dt / 1.0e9, n)
        self.B = self.A[:, 3:6] / m
        self.x = x0
        self.y = x0 + self.sv * np.random.normal((6, ))

    @property
    def mean_motion(self):
        return self.n

    @property
    def measured_dr(self):
        """Current, measured position of the follower spacecraft in the HILL
        frame.
        """
        return self.y[0:3]

    @property
    def measured_dv(self):
        """Current, measured velocity of the follower spacecraft in the HILL
        frame.
        """
        return self.y[3:6]

    @property
    def time_ns(self):
        """Current simulation time in nanoseconds.
        """
        return self.t_ns

    @property
    def time_s(self):
        """Current simulation time in seconds.
        """
        return self.t_ns / 1.0e9

    @property
    def true_dr(self):
        """Current position of the follower spacecraft in the HILL frame.
        """
        return self.x[0:3]

    @property
    def true_dv(self):
        """Current velocity of the follower spacecraft in the HILL frame.
        """
        return self.x[3:6]

    def run(self, u=None):
        super(ClohessyWiltshireDynamics, self).run()

        self.x = np.matmul(self.A, self.x) + self.sw * np.random.normal((6, ))
        if u is not None:
            self.x = self.x + np.matmul(self.B, u)

        self.y = x0 + self.sv * np.random.normal((6, ))
