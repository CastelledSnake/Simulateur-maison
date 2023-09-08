from math import log10
from random import Random


def original(rng: Random) -> int:
    """
    Returns a digit in base 10 according to a Benford distribution.
    :param rng: A Random instance.
    :return: An int.
    """
    prob = rng.random()  # Pick a position on the [0.0, 1.0) segment, following a uniform distribution.
    k = 1
    cursor = log10(1 + 1 / k)  # Cursor that increases on the [0.0, 1.0) axis.
    while cursor <= prob:  # While the cursor hasn't met the position picked up, the number to return increases.
        k += 1
        cursor += log10(1 + 1 / k)
    return k


def benford(position: int, rng: Random) -> int:
    """
    Computes a base 10 digit according to Benford law,
    with the approximation of uniform distribution for digits at position superior to the 1st.
    :param position: The position that the returned number will have in the final number.
    :param rng: A Random instance.
    :return: An int.
    """
    if position == 1:
        return original(rng)
    elif position > 1:
        return rng.randrange(10)
