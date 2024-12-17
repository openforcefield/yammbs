"""Code vendored from ForceBalance."""

import numpy
from numpy.typing import NDArray


def periodic_diff(
    a: NDArray[numpy.float64],
    b: NDArray[numpy.float64],
    v_periodic: float,
) -> NDArray[numpy.float64]:
    """
    Convenience function for computing the minimum difference in periodic coordinates

    Parameters
    ----------
    a
        Reference values in a numpy array
    b
        Target values in a numpy arrary
    v_periodic
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


def compute_rmsd(
    ref: NDArray[numpy.float64],
    tar: NDArray[numpy.float64],
    v_periodic: float | None = None,
) -> float:
    """
    Compute the RMSD between two arrays, supporting periodic difference
    """

    assert len(ref) == len(tar), "array length must match"

    if len(ref) == 0:
        return 0.0

    diff = ref - tar if v_periodic is None else periodic_diff(ref, tar, v_periodic)

    return numpy.sqrt(numpy.sum(diff**2) / len(ref))
