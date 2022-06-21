""" This file has functions for validating input data etc. """

import logging

from .classes import Trip
from .exceptions import MultipleDaysNotSupported

# Initialize logger.
log = logging.getLogger(__name__)


def check_trips_only_has_single_date(trips: list[Trip]) -> None:
    """
    Checks if only one single start date is present in the list, or else raises an MultipleDaysNotSupported.
    :param trips: Trips.
    """
    # Check that all trips are from a single day in the sense that they all start on a single day, otherwise throw exception.
    dates = {trip.start_time.date() for trip in trips}

    if len(dates) > 1:
        log.warning("Multiple start days found, when trying to optimize a single day.")
        raise MultipleDaysNotSupported(
            "Multiple start days found, when trying to optimize a single day."
        )
