
class Dynamics(object):
    """Defines an interface all dynamics models must adhere to.
    """
    def __init__(self, dt):
        super(Dynamics, self).__init__()

        self.dt_ns = dt
        self.t_ns = 0

    @property
    def mean_motion(self):
        """Current mean motion to be used for Clohessy-Wiltshire dynamics
        generation.
        """
        raise NotImplementedError()

    @property
    def measured_dr(self):
        """Current, measured position of the follower spacecraft in the HILL
        frame.
        """
        raise NotImplementedError()

    @property
    def measured_dv(self):
        """Current, measured velocity of the follower spacecraft in the HILL
        frame.
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    @property
    def true_dv(self):
        """Current velocity of the follower spacecraft in the HILL frame.
        """
        raise NotImplementedError()

    def step(self):
        """Steps the simulation time forward. Derived classes should implement
        the actual dynamics of this model as well.
        """
        self.t_ns += self.dt_ns
