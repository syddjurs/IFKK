""" This file defines functions for generating random test instances."""
import datetime
import math
import random

from .classes import Trip, Trips


def generate_trips(
    earliest_start_hour: int,
    latest_start_hour: int,
    number_of_trips: int,
    minimum_length_in_kilometers: float,
    maximum_length_in_kilometers: float,
) -> Trips:
    """
    Generates the specified number of trips with a starting time within the earliest start hour and latest start hour within the specified range in kilometers.

    In general a uniform distribution is applied to the different fields.

    The trips will be generated to start at minute 00/60, 15, 30 or 45.

    :param earliest_start_hour: E.g. 6 meaning, the earliest start time would be 6:00. Must be >= 0.
    :param latest_start_hour: E.g. 23 meaning, the latest start time would be 23:00. Must be <= 23.
    :param number_of_trips: E.g. 100 meaning, 100 trips are generated.
    :param minimum_length_in_kilometers: The minimum length of a trip.
    :param maximum_length_in_kilometers: The maximum length of a trip.
    :return: A class of Trips.
    """

    trips_generated = []

    for i in range(number_of_trips):
        trip_id = i
        # Round the length to two decimals to make things easier to read and debug.
        length = round(
            random.uniform(minimum_length_in_kilometers, maximum_length_in_kilometers),
            2,
        )
        # Start time contains a hardcoded year, hardcoded month, hardcoded day, hours 00, 15, 30 or 45 set in a pseudo-randomized manner and minutes 00, 15, 30 or 45 set in a pseudo-randomized manner.
        start_time = datetime.datetime(
            year=2020,
            month=1,
            day=1,
            hour=random.randint(earliest_start_hour, latest_start_hour + 1),
            minute=15 * random.randint(0, 3),
        )
        # The end time is the start_time + 15, 30, 45, 60, 75 or 90 (set pseudo-randomly), minutes added. Must be converted to hours and minutes, as timedelta only counts minutes to 59.
        minutes_total = 15 * random.randint(1, 7)
        hours = math.floor(minutes_total / 60)
        minutes = minutes_total - 60 * hours
        end_time = start_time + datetime.timedelta(hours=hours, minutes=minutes)

        trips_generated.append(
            Trip(
                id=trip_id,
                start_time=start_time,
                end_time=end_time,
                length_in_kilometers=length,
            )
        )

    return Trips(trips=trips_generated)


if __name__ == "__main__":

    number_of_trips_generated = 300

    # Generate some trips.
    trips = generate_trips(6, 18, number_of_trips_generated, 1, 30)

    # Dump the file here.
    with open(f"trips_{number_of_trips_generated}.json", "w") as file:
        file.write(trips.json())
