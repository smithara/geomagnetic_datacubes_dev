"""
"""

import numpy as np


def rtp2NEC(B_rtp):
    """Change an array from rtp to NEC frame."""
    if B_rtp.shape[1] != 3:
        raise Exception("B_rtp is not the right shape")
    return np.stack((-B_rtp[:, 1], B_rtp[:, 2], -B_rtp[:, 0])).T


def NEC2rtp(B_NEC):
    """Change an array from NEC to rtp frame."""
    if B_NEC.shape[1] != 3:
        raise Exception("B_NEC is not the right shape")
    return np.stack((-B_NEC[:, 2], -B_NEC[:, 0], B_NEC[:, 1])).T


def sph2cart(R, t, p):
    """Spherical to Cartesian conversion.

    Args:
        R (ndarray): radius
        t (ndarray): theta, colatitude in degrees (0<t<180)
        p (ndarray): phi, longitude, in degrees (0<p<360)

    Returns:
        X (ndarray): Coordinates in GEO frame
        Y (ndarray):
        Z (ndarray):

    """
    # Calculate the sines and cosines
    rad = np.pi/180  # conversion factor, degrees to radians
    s_p = np.sin(p*rad)
    s_t = np.sin(t*rad)
    c_p = np.cos(p*rad)
    c_t = np.cos(t*rad)
    # Calculate the x,y,z over the whole grid
    X = R*c_p*s_t
    Y = R*s_p*s_t
    Z = R*c_t
    return X, Y, Z
