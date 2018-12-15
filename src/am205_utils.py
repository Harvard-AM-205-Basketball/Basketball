"""
Harvard IACS Applied Math 205
Utilites

Michael S. Emanuel
Tue Sep 11 16:36:40 2018
"""

import numpy as np
from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import Callable, Tuple

# Type aliases
funcType = Callable[[float], float]


def plot_style() -> None:
    """Set plot style for the session."""
    # Set up math plot library to use TeX
    # https://matplotlib.org/users/usetex.html
    plt.rc('text', usetex=True)
    # Set default font size to 20
    mpl.rcParams.update({'font.size': 30})


# *************************************************************************************************
def range_inc(x: int, y: int = None, z: int = None) -> range:
    """Return a range inclusive of the end point, i.e. range(start, stop + 1, step)"""
    if y is None:
        (start, stop, step) = (1, x + 1, 1)
    elif z is None:
        (start, stop, step) = (x, y + 1, 1)
    elif z > 0:
        (start, stop, step) = (x, y + 1, z)
    elif z < 0:
        (start, stop, step) = (x, y - 1, z)
    return range(start, stop, step)


def arange_inc(x: float, y: float = None, z: float = None) -> np.ndarray:
    """Return a numpy arange inclusive of the end point, i.e. range(start, stop + 1, step)"""
    if y is None:
        (start, stop, step) = (1, x + 1, 1)
    elif z is None:
        (start, stop, step) = (x, y + 1, 1)
    elif z > 0:
        (start, stop, step) = (x, y + z, z)
    elif z < 0:
        (start, stop, step) = (x, y - z, z)
    return np.arange(start, stop, step)


def quadRootPos(a: float, b: float, c: float) -> float:
    """Return positive root of quadratic as a float; discriminant has been prechecked."""
    # Flip sign of a if necessary so a > 0
    if a < 0:
        (a, b, c) = (-a, -b, -c)
    disc: float = b * b - 4 * a * c
    sd: float = sqrt(disc)
    return (-b + sd) / (2*a)


def quadRootNeg(a: float, b: float, c: float) -> float:
    """Return negative root of quadratic as a float; discriminant has been prechecked."""
    # Flip sign of a if necessary so a > 0
    if a < 0:
        (a, b, c) = (-a, -b, -c)
    disc: float = b * b - 4 * a * c
    sd: float = sqrt(disc)
    return (-b - sd) / (2*a)


def quadRoots(a: float, b: float, c: float) -> Tuple[complex, complex]:
    """Return both roots of a quadratic"""
    # Flip sign of a if necessary so a > 0
    if a < 0:
        (a, b, c) = (-a, -b, -c)
    # Calculate the discriminant
    disc: float = b * b - 4 * a * c
    # If the discriminant is non-negative, return two real rots
    if disc >= 0:
        sd: float = sqrt(disc)
    else:
        sd: complex = sqrt(-disc) * 1j
    r1 = (-b - sd) / (2*a)
    r2 = (-b + sd) / (2*a)
    return (r1, r2)


# *************************************************************************************************
def rms(x: np.ndarray) -> float:
    """Root mean square of array x"""
    return np.sqrt(np.mean(x*x))


# *************************************************************************************************
def mat2tex_row(r, fmt: str = '0.3f'):
    """Generate a row in the body of a matrix"""
    # Alias special characters to variables for legibility
    line_end = r'\\'
    # Basic unformatted stub for one entry
    entry_unfmt = '{0:fmt}'.replace('fmt', fmt)
    # List of formatted entries
    entries = [entry_unfmt.format(x) for x in r]
    # Join the entries into one row string and append a \\ to end the line
    row_str = ' & '.join(entries) + f' {line_end}'
    return row_str


def mat2tex(A, fmt: str = '0.3f'):
    """Convert the mxn matrix A to LaTeX format."""
    # The header and footer row - raw strings
    col_header = 'c' * A.shape[1]
    A_str_header: str = r'\left[ \begin{array}{<col_header>}'
    A_str_header = A_str_header.replace('<col_header>', col_header)
    A_str_footer: str = r'\end{array}\right]'
    # The rows of A
    A_str_rows: str = [mat2tex_row(r, fmt) for r in A.tolist()]
    # Get rid of trailing \\ in the last row
    line_end = r'\\'
    A_str_rows[-1].replace(f' {line_end}', '')
    # Append the header and footer
    A_str_rows = [A_str_header] + A_str_rows + [A_str_footer]
    # Assemble the matrix string by joining the rows
    A_str = '\n'.join(A_str_rows)
    return A_str
