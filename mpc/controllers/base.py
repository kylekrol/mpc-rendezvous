import cvxpy as cp


class Controller(object):
    """Defines an interface all controllers must adhere to. The properties
    defined here should only be accessed once the controller has been run.
    """
    @property
    def control(self):
        """Most recent impulse requested by the controller.
        """
        raise NotImplementedError()

    @property
    def total_cost(self):
        """Most recent optimal cost calculated by the controller. This is
        the total of the stage and terminal costs.
        """
        raise NotImplementedError()

    @property
    def stage_costs(self):
        """Most recent optimal costs calculated by the controller. This is the
        cost "trajectory" over time.
        """
        raise NotImplementedError()

    @property
    def horizon(self):
        """Most recent optimal horizon calculated by the controller. For a fixed
        horizon controller this doesn't change.
        """
        raise NotImplementedError()

    @property
    def status(self):
        """Most recent status reported by the optimal solver.
        """
        raise NotImplementedError()

    @property
    def trajectory(self):
        """Most recent optimal state and control trajectory calculated by the
        controller. This returns a tuple.
        """
        raise NotImplementedError()

    def run(self, A, B, x0, verbose=False):
        """Runs the controller given the updated dynamics and initial state.
        Returns true if successful.
        """
        raise NotImplementedError()


class FixedHorizonController(Controller):
    """Defines a starting point for all controllers using a fixed prediction
    horizon. This class can be extended to define more specific forms of the
    problem.
    """
    def __init__(self, N, J, *args):
        super(FixedHorizonController, self).__init__()

        # System dynamics matrices and horizon
        self.N = N
        self.A = cp.Parameter((6, 6), name='A')
        self.B = cp.Parameter((6, 3), name='B')

        # Initial state
        self.x0 = cp.Parameter((6, ), name='x0')

        # Decision variables
        self.X = cp.Variable((6, N + 1), name='X')
        self.U = cp.Variable((3, N), name='U')

        # Initialize stage costs to the provided constant and final stage cost
        # to zero
        self.costs = [J for i in range(N)]
        self.costs.append(0)

        # Initialize constraints for dynamics
        self.constraints = [self.X[:, 0] == self.x0]
        for i in range(N):
            self.constraints += [
                self.X[:, i + 1] == self.A @ self.X[:, i] + self.B @ self.U[:, i]
            ]

        self._add_to_constaints_and_costs(*args)

        # Assemble final objective function
        self.objective = 0;
        for i in range(N + 1):
            self.objective += self.costs[i]

        # Full problem
        self.problem = cp.Problem(cp.Minimize(self.objective), self.constraints)

    def _add_to_constaints_and_costs(self, *args):
        """Allows a base class to update the constraints and objective function
        of the convex optimization problem prior to constructing the final
        problem.
        """
        pass

    @property
    def control(self):
        return self.U[:, 0].value

    @property
    def total_cost(self):
        return self.objective.value

    @property
    def stage_costs(self):
        return [cost.value for cost in self.costs]

    @property
    def horizon(self):
        return self.N

    @property
    def status(self):
        return self.problem.status

    @property
    def trajectory(self):
        return self.X.value, self.U.value

    def run(self, A, B, x0, verbose=False):
        """Runs the fixed horizon controller given updated dynamics and an
        initial state. Returns true if the solver was successful.
        """
        self.A.value = A
        self.B.value = B
        self.x0.value = x0

        self.problem.solve(verbose=verbose)

        # Warn on optimal inaccurate
        if self.problem.status == cp.OPTIMAL_INACCURATE:
            print("WARNING: Optimal inaccurate solution obtained.")

        return self.problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


class VariableHorizonController(Controller):
    """Defines an interface for variable horizon controllers of any form.
    Essentially, this wraps around multiple controllers with different horizons
    and then just optimizes over the horizon length along with the other
    decision variables.
    """
    def __init__(self, N, Controller, *args):
        super(VariableHorizonController, self).__init__()

        self.controllers = [(i, Controller(i, *args)) for i in range(1, N + 1)]
        self.optimal_controller = None

        # Maximum horizon
        self.N = N

    @property
    def control(self):
        return self.optimal_controller.control

    @property
    def total_cost(self):
        return self.optimal_controller.total_cost

    @property
    def stage_costs(self):
        return self.optimal_controller.stage_costs

    @property
    def horizon(self):
        return self.optimal_controller.horizon

    @property
    def status(self):
        return self.optimal_controller.status

    @property
    def trajectory(self):
        return self.optimal_controller.X.value, self.optimal_controller.U.value

    def run(self, A, B, x0, verbose=False):
        """Runs all fixed horizon controllers and minimizes over horizon size.
        Returns true if an optimal horizon is found.
        """
        self.optimal_controller = None
        for _, controller in self.controllers:
            if verbose:
                print("Solving for a horizon of {} / {}".format(_, self.N))

            try:
                controller.run(A, B, x0)
            except cp.error.SolverError:
                continue

            # Skip the controller if it doesn't give good results
            if controller.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
                continue

            # We'll take if if we don't have one already
            if self.optimal_controller is None:
                self.optimal_controller = controller
            # Otherwise we need to perform some comparisons
            else:
                if self.optimal_controller.status == controller.status:
                    if controller.objective.value < self.optimal_controller.objective.value:
                        self.optimal_controller = controller
                elif self.optimal_controller.status == cp.OPTIMAL_INACCURATE:
                    self.optimal_controller = controller

        return self.optimal_controller is not None
