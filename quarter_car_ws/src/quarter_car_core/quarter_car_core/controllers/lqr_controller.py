"""
LQR Controller Stub
===================
References: [5] Hrovat 1997, [1] MathWorks state-space
State-space matrices from QuarterCarODE.get_state_space().
Use scipy.linalg.solve_discrete_are.
Q = diag([1000, 10, 100, 10]), R = [[0.01]]
"""


class LQRController:
    def __init__(self, params: dict = None):
        raise NotImplementedError("LQR not implemented — stub only")

    def predict(self, obs):
        raise NotImplementedError("LQR not implemented — stub only")
