def pretty_print(value: int or float, unit: str, round_size: int = 3):
    """
    Pretty-prints a value with adequate unit, multiple and size.
    :param value: The numerical value to be printed (int of float)
    :param unit: The physical unit of value (str)
    :param round_size: The limit to round the number to print
    :return: A nice string of the value with its correct multiple and unit (str)
    """
    units = [(1e30, "Q"), (1e27, "R"), (1e24, "Y"), (1e21, "Z"), (1e18, "E"), (1e15, "P"), (1e12, "T"), (1e9, "G"),
             (1e6, "M"), (1e3, "k"), (1e2, "h"), (1e1, "da"), (1e0, ""), (1e-1, "d"), (1e-2, "c"), (1e-3, "m"),
             (1e-6, "µ"), (1e-9, "n"), (1e-12, "p"), (1e-15, "f"), (1e-18, "a"), (1e-21, "z"), (1e-24, "y"),
             (1e-27, "r"), (1e-30, "q")]
    for threshold, unit_prefix in units:
        if value >= threshold:
            return f'{round(value / threshold, round_size)} {unit_prefix}{unit}'
