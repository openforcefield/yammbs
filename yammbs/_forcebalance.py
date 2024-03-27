"""Code vendored from ForceBalance."""
import numpy


def periodic_diff(a, b, v_periodic):
    """convenient function for computing the minimum difference in periodic coordinates
    Parameters
    ----------
    a: np.ndarray or float
        Reference values in a numpy array
    b: np.ndarray or float
        Target values in a numpy arrary
    v_periodic: float > 0
        Value of the periodic boundary

    Returns
    -------
    diff: np.ndarray
        The array of same shape containing the difference between a and b
        All return values are in range [-v_periodic/2, v_periodic/2),
        "( )" means exclusive, "[ ]" means inclusive

    Examples
    -------
    periodic_diff(0.0, 2.1, 2.0) => -0.1
    periodic_diff(0.0, 1.9, 2.0) => 0.1
    periodic_diff(0.0, 1.0, 2.0) => -1.0
    periodic_diff(1.0, 0.0, 2.0) => -1.0
    periodic_diff(1.0, 0.1, 2.0) => -0.9
    periodic_diff(1.0, 10.1, 2.0) => 0.9
    periodic_diff(1.0, 9.9, 2.0) => -0.9
    """
    assert v_periodic > 0
    h = 0.5 * v_periodic
    return (a - b + h) % v_periodic - h


def compute_rmsd(ref, tar, v_periodic=None):
    """
    Compute the RMSD between two arrays, supporting periodic difference
    """
    assert len(ref) == len(tar), "array length must match"
    n = len(ref)
    if n == 0:
        return 0.0
    if v_periodic is not None:
        diff = periodic_diff(ref, tar, v_periodic)
    else:
        diff = ref - tar
    rmsd = numpy.sqrt(numpy.sum(diff**2) / n)
    return rmsd
