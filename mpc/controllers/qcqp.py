from .base import FixedHorizonController

import cvxpy as cp


class QCQPController(FixedHorizonController):
    """Rendezvous controller built around a quadratically constrained quadratic
    program.
    """
    def __init__(self, N, J, Q, R, rho, umax):
        super(QCQPController, self).__init__(N, J, Q, R, rho, umax)

    def _add_to_constaints_and_costs(self, Q, R, rho, umax):
        """Add final stage quadratic stage cost, L1 + L2 stage control costs,
        and quadratic stage control constraints.
        """
        for i in range(self.N):
            self.constraints.append(cp.norm(self.U[:, i]) <= umax)

        for i in range(self.N):
            self.costs[i] += cp.quad_form(self.U[:, i], R) + rho * cp.norm(self.U[:, i], p=1)
        self.costs[self.N] += cp.quad_form(self.X[:, self.N], Q)
