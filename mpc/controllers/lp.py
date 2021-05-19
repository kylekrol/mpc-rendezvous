from .base import FixedHorizonController

import cvxpy as cp
import numpy as np


class LPController(FixedHorizonController):
    """Rendezvous controller built around a linear program.
    """
    def __init__(self, N, J, rho, rtol, vtol, umax):
        super(LPController, self).__init__(N, J, rho, rtol, vtol, umax)

    def _add_to_constaints_and_costs(self, rho, rtol, vtol, umax):
        """Add final stage affine constraints, control constraints, and L1 stage
        control costs.
        """
        self.constraints.append(cp.abs(self.X[0:3, self.N]) <= rtol)
        self.constraints.append(cp.abs(self.X[3:6, self.N]) <= vtol)
        for i in range(self.N):
            self.constraints.append(cp.abs(self.U[:, i]) <= umax / np.sqrt(3))

        for i in range(self.N):
            self.costs[i] += rho * cp.norm(self.U[:, i], p=1)

    @property
    def stage_costs(self):
        return [cost.value for cost in self.costs[:-1]]
