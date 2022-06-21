""" This file defines a number of custom exceptions."""


class MultipleDaysNotSupported(Exception):
    """Thrown, when multiple days are not supported (usually in the optimization)."""

    pass


class UnsupportedAlgorithmType(Exception):
    """Thrown, when the provided algorithm is not supported."""

    pass


class NumberOfTripsPlannedMismatchException(Exception):
    """Thrown, when number of trips does not match number of trips assigned."""

    pass


class NoSolutionFoundException(Exception):
    """Thrown, when no solution can be found."""

    pass


class RangeExceededException(Exception):
    """Thrown, when a vehicle is assigned more kilometers than its range."""

    pass
