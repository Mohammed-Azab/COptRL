"""
MPC Controller Stub
===================
References: [1] MathWorks ADMM-MPC, [8] Boyd ADMM, [9] OSQP
Prediction horizon N=20, input constraints +-10000 N
QP solver: osqp (pip install osqp)
See: https://de.mathworks.com/help/mpc/ug/admm-based-mpc-control-for-quarter-car-suspension.html
"""


class MPCController:
    def __init__(self, params: dict = None):
        raise NotImplementedError("MPC not implemented — stub only")

    def predict(self, obs):
        raise NotImplementedError("MPC not implemented — stub only")
