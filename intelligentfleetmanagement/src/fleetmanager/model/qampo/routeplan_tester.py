""" This file holds basic information for testing a solution for the small trips and 3 vehicles problem."""

import logging

from exceptions import (
    MultipleDaysNotSupported,
    NumberOfTripsPlannedMismatchException,
    RangeExceededException,
)

from .classes import RoutePlan

# Initialize logger.
log = logging.getLogger(__name__)


def check_solution_for_problem(number_of_trips: int, solution: RoutePlan) -> None:
    """
    This function performs a number of checks to find possible issues and raises exceptions if any exists.
    @param number_of_trips Number of trips.
    @param solution The solution to be checked for issues.
    """
    # Assert that all vehicles have trips, and the sum of all trips is equals to number_of_trips.
    trips_assigned = 0

    for assignment in solution.assignments:
        trips_assigned += len(assignment.route.trips)

        # Check that the trips for this vehicle are non-overlapping (not for employee car).
        if len(assignment.route.trips) > 1:
            # Initialize a running start time and end time.
            start_time = assignment.route.trips[0].start_time
            end_time = assignment.route.trips[0].end_time

            # Iterate over the remaining trips and check for overlap.
            # Assume that start_time <= end_time for all trips and increasing start_time.
            for trip in assignment.route.trips[1:]:
                # Check if there is an overlap.
                # See https://stackoverflow.com/questions/325933/determine-whether-two-date-ranges-overlap
                if max(start_time, trip.start_time) < min(end_time, trip.end_time):

                    # Overlap.
                    log.debug(
                        "An overlap between trips has been found. Multiple days are not supported."
                    )
                    raise MultipleDaysNotSupported

                start_time = trip.start_time
                end_time = trip.end_time
        # Check the total range.
        total_length = sum(trip.length_in_kilometers for trip in assignment.route.trips)

        # Check that the range is not longer than the allowed range.
        if total_length > assignment.vehicle.range_in_kilometers:

            log.debug("Total range of trips has exceeded the allowed range.")
            raise RangeExceededException

    # Count the number of trips for the employee car.
    trips_assigned += len(solution.employee_car.route.trips)

    # Check if the number of trips assigned is not equal to the number of trips.
    if not trips_assigned == number_of_trips:

        log.debug("The number of trips assigned is not equal to the number of trips.")
        raise NumberOfTripsPlannedMismatchException
