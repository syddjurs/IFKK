""" This file contains code that optimizes a single day routing given a fixed fleet in a greedy manner. This is conceptually
the same as the algorithm in the overall project (IFFK) that simulates car assignment previously."""

import copy
import datetime
import logging

from .classes import BaseVehicle, RoutePlan, RoutingAlgorithm, Trip, Vehicle
from .exceptions import NoSolutionFoundException
from .helper_functions import prioritize_vehicles_according_to_weighted_variable_costs
from .routeplan_factory import route_plan_from_vehicle_trip_map
from .validation import check_trips_only_has_single_date

# Initialize logger.
log = logging.getLogger(__name__)


class RoutingGreedy(RoutingAlgorithm):
    def optimize_single_day(
        self,
        trips: list[Trip],
        vehicles: list[Vehicle],
        employee_car: BaseVehicle,
        emission_cost_per_ton_co2: float = 1500,
        time_limit_in_seconds: int = 60,
    ) -> RoutePlan:
        """
        This is a simple greedy algorithm that assigns trips to vehicles based on a weight of the variable cost per kilometer and the CO2 emission. Greedily means that earliest trips are assigned first to the cheapest vehicle. time_limit_in_seconds specifies for how long, the algorithm is allowed to be run.
        :param trips: List of trips in the route plan.
        :param vehicles: List of vehicles in the route plan.
        :param employee_car: Employee car a trip can be assigned to.
        :param emission_cost_per_ton_co2: CO2 emission cost per ton for the entire route plan.
        :param time_limit_in_seconds: Time limit for the running time of the algorithm.
        :return Routing plan created after optimization has been performed.
        """

        stop = datetime.datetime.now() + datetime.timedelta(
            seconds=time_limit_in_seconds
        )
        log.info("Running greedy single day optimization.")

        # Check the trips. If any issues occur, throw an exception.
        check_trips_only_has_single_date(trips)

        log.debug("Checked trips input for being single day.")

        # Sort by earliest start time.
        trips_sorted = copy.deepcopy(trips)
        trips_sorted.sort(key=lambda t: t.start_time)

        log.debug("Sorted trips.")

        if datetime.datetime.now() >= stop:
            raise NoSolutionFoundException()

        # Store the time at which a car becomes available. Initialized at midnight on the date of the trips.
        midnight_date = datetime.datetime.combine(
            trips_sorted[0].start_time,
            datetime.datetime.strptime("0000", "%H%M").time(),
            tzinfo=trips_sorted[0].start_time.tzinfo,
        )

        # Note, this is not done for the employee car. Employee car is always available.
        available_at = {vehicle: midnight_date for vehicle in vehicles}

        # Store how much range is left for each vehicle. Note, this is not done for the employee car. It is always possible to assign more trips to an employee car.
        range_left = {vehicle: vehicle.range_in_kilometers for vehicle in vehicles}

        # Store how much "up time" is left for each vehicle. Note, this is not done for the employee car. It is always possible to assign more trips to an employee car.
        uptime_left = {
            vehicle: vehicle.maximum_driving_in_minutes for vehicle in vehicles
        }

        # Store for each vehicle and employee car the trips assigned to it. Initialized with an empty list.
        assignments: {Vehicle: list} = {vehicle: [] for vehicle in vehicles}
        assignments.update({employee_car: []})

        if datetime.datetime.now() >= stop:
            raise NoSolutionFoundException()

        # Get a prioritized list of vehicles, including the employee car, by weighting the CO2 emission with the variable cost.
        vehicles_prioritized = prioritize_vehicles_according_to_weighted_variable_costs(
            vehicles + [employee_car], emission_cost_per_ton_co2
        )

        log.debug("Starting greedy assignment of trips to vehicles.")

        # Assign trips to vehicles in a greedy fashion.
        for trip in trips_sorted:

            if datetime.datetime.now() >= stop:
                raise NoSolutionFoundException()

            # Assign to the vehicles in a greedy fashion if possible.
            for vehicle in vehicles_prioritized:

                # Check if the vehicle is available and has ample range + "up time" left, except for the employee car option, which is always available.
                if (
                    vehicle in available_at
                    and available_at.get(vehicle) <= trip.start_time
                    and range_left.get(vehicle) >= trip.length_in_kilometers
                    and uptime_left.get(vehicle) >= trip.get_trip_length_in_minutes()
                ):
                    # If this is not the employee car, update the available time.
                    available_at.update({vehicle: trip.end_time})
                    # Update the range left.
                    range_left.update(
                        {vehicle: range_left.get(vehicle) - trip.length_in_kilometers}
                    )
                    # Update the uptime left.
                    uptime_left.update(
                        {
                            vehicle: uptime_left.get(vehicle)
                            - trip.get_trip_length_in_minutes()
                        }
                    )
                elif vehicle in available_at:
                    # The vehicle cannot serve the trip due to overlap.
                    continue

                # Assign trip to vehicle.
                assignments.get(vehicle).append(trip)

                # The trip is assigned a vehicle, and stop the inner loop.
                break

        log.info("Greedy single day optimization done.")
        # Create a Route plan from the assignments.
        return route_plan_from_vehicle_trip_map(vehicle_assignments=assignments)
