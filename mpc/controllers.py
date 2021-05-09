import cvxpy as cp


class Controller(object):

    def __init__(self):
        super(Controller, self).__init__()

        self.__has_run = False
        self._control = None
        self._cost = None
        self._horizon = None
        self._trajectory = None, None

    @property
    def control(self):
        """Provides the most recent control requested by the controller.
        """
        if not self.__has_run:
            raise RuntimeError("The controller must be run before requesting the control.")

        return self._control

    @property
    def cost(self):
        """Provides the most recent objective cost requested by the controller.
        """
        if not self.__has_run:
            raise RuntimeError("The controller must be run before requesting the cost.")

        return self._cost

    @property
    def horizon(self):
        """Provides the most recent time horizon utilized by the controller.
        """
        if not self.__has_run:
            raise RuntimeError("The controller must be run before requesting the horizon.")

        return self._horizon

    @property
    def trajectory(self):
        """Provides the most recent state and control trajectories calculated
        by the controller.
        """
        if not self.__has_run:
            raise RuntimeError("The controller must be run before requesting the trajectory.")

        return self._trajectory

    def run(self):
        self.__has_run = True


class FixedHorizonController(Controller):

    def __init__(self, N, Q, R, rho, umax):
        super(FixedHorizonController, self).__init__()

        self._horizon = N

        # System dynamics matrices
        self.A = cp.Parameter((6, 6), name='A')
        self.B = cp.Parameter((6, 3), name='B')

        # Initial state
        self.x0 = cp.Parameter((6, ), name='x0')

        # Decision variables
        self.X = cp.Variable((6, N + 1), name='X')
        self.U = cp.Variable((3, N), name='U')

        # Objective function
        self.objective = cp.quad_form(self.X[:, N], Q)
        for i in range(N):
            self.objective = cp.quad_form(self.U[:, i], R) + rho * cp.norm(self.U[:, i], p=1)

        # Constraints
        constraints = [self.X[:, 0] == self.x0]
        for i in range(N):
            constraints += [
                self.X[:, i + 1] == self.A @ self.X[:, i] + self.B @ self.U[:, i],
                cp.norm(self.U[:, i]) <= umax
            ]

        # Full problem
        self.problem = cp.Problem(cp.Minimize(self.objective), constraints)

    def run(self, A, B, dr, dv):
        super(FixedHorizonController, self).run()

        self.A.value = A
        self.B.value = B
        self.x0.value = [dr[0], dr[1], dr[2], dv[0], dv[1], dv[2]]

        self.problem.solve()

        self._control = self.U.value[:, 0]
        self._cost = self.objective.value
        self._trajectory = self.X.value, self.U.value

        return self.problem.status == cp.OPTIMAL


class VariableHorizonController(Controller):

    def __init__(self, N, J, Q, R, rho, umax):
        super(VariableHorizonController, self).__init__()

        self.controllers = [
            (i, FixedHorizonController(i, Q, R, rho, umax)) for i in range(1, N)
        ]

        self.J = J
        self.controller = None
        self.objective = None

    def run(self, A, B, dr, dv):
        super(VariableHorizonController, self).run()

        optimal = None
        optimal_horizon = None
        optimal_objective = -1

        for N, controller in self.controllers:
            controller.run(A, B, dr, dv)
            objective = controller.objective.value + N * self.J

            if controller.problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                continue

#            print(N, objective)

            if optimal_objective < 0 or objective < optimal_objective:
                horizon = N
                optimal = controller
                optimal_horizon = N
                optimal_objective = objective

#        assert False

        print(horizon)

        self.controller = optimal
        self.objective = optimal_objective

        self._control = optimal.U.value[:, 0]
        self._cost = optimal_objective
        self._horizon = optimal_horizon
        self._trajectory = optimal.X.value, optimal.U.value

        return self.controller.problem.status == cp.OPTIMAL
