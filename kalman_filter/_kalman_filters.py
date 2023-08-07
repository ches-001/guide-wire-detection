import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class KalmanFilter:
    r"""
    The Kalman filter addresses the general problem of trying to estimate the state
    of a discrete-time controlled process that is governed by a linear stochastic
    difference equation

    A ndarray[N, N]:
        state transition square matrix
    Q ndarray[N, N]:
        process noise covariance
    H ndarray[M, N]:
        Measurement square matrix (for mapping state space to measurement)
    R ndarray[M, M]:
        Measurement covarience matrix (uncertainties in measurement)
    B ndarray[N, 1]:
        control input transition matrix
    """
    A: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    R: np.ndarray
    B: Optional[np.ndarray] = None

    def estimate(self, X: np.ndarray, P: np.ndarray, U: Optional[np.ndarray]=None) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Parameters:
        -------------------------------------
        X ndarray[N, 1] :
            predicted state / mean at (t-1)
        P ndarray[N, N]:
            convariance of state at (t-1)
        U: Optional[ndarray[N, 1]]:
            control input of system

        Equations:
        -------------------------------------
        Linear Stochastic Difference Equations:
            X := A X + B U

            P := (A P A^T) + Q

        Returns:
        -------------------------------------
        X ndarray[N, 1]
            estimated state / mean at (t)

        X ndarray[N, N]
            estimated estimate covariance at (t)
        """
        X = self.A @ X
        if (U is not None) and (self.B is not None):
            X += self.B @ U
        P = (self.A @ P @ self.A.T) + self.Q
        return X, P

        

    def update(self, X: np.ndarray, Y: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Parameters:
        -------------------------------------
        X ndarray[N, 1]:
            predicted state / mean at (t)
        X ndarray[N, 1]:
            estimated state / mean at (t)
        P ndarray[N, N]:
            convariance of state at (t)

        Equations:
        -------------------------------------
        V = Y - H X                     (measurement residual)

        S = (H P H^T) + R               (measurement prediction covariance)

        K = P H^T inv(S)                (kalman gain)

        X := X + K V                    (updated / corrected X value)

        P := P - K S K^T                (updated / corrected P value)

        Returns:
        -------------------------------------
        X ndarray[N, 1]
            updated state / mean at (t)

        X ndarray[N, N]
            updated estimate covariance at (t)
        """
        V = Y - (self.H @ X)
        S = (self.H @ P @ self.H.T) + self.R
        K = P @ self.H.T @ np.linalg.inv(S)
        X = X + (K @ V)
        P = P - (K @ S @ K.T)
        return X, P