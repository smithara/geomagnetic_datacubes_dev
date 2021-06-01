"""
"""

import numpy as np
import pandas as pd
import chaosmagpy as cp


def write_shc(n=None, m=None, gh=None, shc_file=None):
    """Write an SHC-format file given a set of coefficients.

    Write an shc file from mixed gh input 1D array and matching n,m arrays
    ( gh = [g01, g11, h11, g02, g12, h12, g22, h22 ...] )

    CURRENTLY ONLY FOR A FIXED YEAR...

    Args:
        n (array): degree values for the coefficients in gh
        m (array): order values for the coefficients in gh
        gh (array): mixed degree-order list of coefficients
        shc_file (str): full path to the file to be created

    """
    if (n is None) or (m is None):
        lmax = np.sqrt(len(gh) + 1) - 1
        if np.mod(lmax, 1) == 0:
            lmax = int(lmax)
        else:
            raise Exception("gh is wrong length")
        n = 0
        coeffs_l = []
        coeffs_m = []
        for l in range(1, lmax+1):
            coeffs_l.append(l)
            coeffs_m.append(0)
            for m in range(1, l+1):
                coeffs_l.append(l)
                coeffs_m.append(m)
                coeffs_l.append(l)
                coeffs_m.append(-m)
        n = coeffs_l
        m = coeffs_m

    # Parameters for first line of shc file (after the comments)
    N_min = n[0]
    N_max = n[-1]
    spline_order = 1
    # line_1 = '%3d%4d%4d' % (N_min,N_max,spline_order)
    line_1 = '{:3d}{:4d}{:4d}{:4d}{:4d}'.format(
        N_min, N_max, spline_order, 1, 1
    )

    # The second line
    year = 2015.0
    # line_2 = '%19d' % year
    line_2 = '{:19.4f}'.format(year)

    with open(shc_file, 'w') as cf:
        cf.write('# Line for comments...\n')
        cf.write(line_1+'\n')
        cf.write(line_2+'\n')
        for i in range(len(n)):
            if m[i] >= -99:
                cf.write('{:3d}{:4d}{:12.4f}'.format(n[i], m[i], gh[i])+'\n')
            else:
                cf.write('{:3d}{:5d}{:12.4f}'.format(n[i], m[i], gh[i])+'\n')


def read_shc_file(shc_file):
    """Return combined "gh" coefficients from shc file."""
    return cp.data_utils.load_shcfile(shc_file)[1]


def combine_gh(n, m, g, h):
    """Combine full lists of "g" and "h" coefficients into one mixed degree-order list, gh.

    Combine the separated g and h columns into one, with m going to -1 for the h entries
    Original:
    g = [g01, g11, g02, g12, g22 ...]
    h = [h01, h11, h02, h12, h22 ...]
    Combined:
    gh = [g01, g11, h11, g02, g12, h12, g22, h22 ...]
    NB: h0x terms always disappear in the spherical harmonic summations
        so are not included in the gh list

    Args:
        n (array): values for degree
        m (array): values for order
        g (array): "g" coefficients
        h (array): "h" coefficients

    Returns:
        n_out,m_out,gh
        n_out (array): values for degree in the output array
        m_out (array): values for order
        gh (array): mixed degree-order list of coefficients

    """
    n = n.astype(int)
    m = m.astype(int)
    lmax = n[-1]
    length_out = lmax*(lmax+2)  # length of combined gh
    gh = np.empty(length_out, np.float64)
    n_out = np.empty(length_out, int)
    m_out = np.empty(length_out, int)
    i = 0   # g,h entry counter
    j = 0   # gh counter
    for n in range(1, lmax+1):
        m = 0
        n_out[j] = n
        m_out[j] = m
        gh[j] = g[i]
        i += 1
        j += 1
        for m in range(1, n+1):
            n_out[j] = n
            m_out[j] = m
            gh[j] = g[i]
            j += 1
            n_out[j] = n
            m_out[j] = -m
            gh[j] = h[i]
            j += 1
            i += 1
    return n_out, m_out, gh


def read_cof_file(cof_file, headerlength=12, as_shc_order=True):
    """Get coefficients from a cof-format file.

    Read a .cof file and output the n,m, mixed gh arrays (1D)
    gh can then be split into separate g,h arrays with convert_gh

    Args:
        cof_file (str): full path to the file to read
        headerlength (int): number of lines of header (to skip)

    Returns:
        gh (array): mixed degree-order list of coefficients OR:
        cof (Dataframe): of n, m, g, h values

    """
    cof = pd.read_csv(cof_file, skiprows=headerlength, delim_whitespace=True)
    if as_shc_order:
        n, m, gh = combine_gh(*cof.values.T)
        return gh
    else:
        return cof
