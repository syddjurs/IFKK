""" This file contains code, that optimizes a single day routing given a fixed fleet in an optimal manner using the Constraint programming solver from ortools."""

import logging

from ortools.sat.python import cp_model

from .classes import BaseVehicle, RoutePlan, RoutingAlgorithm, Trip, Vehicle
from .cost_calculator import calculate_co2_emission_cost_per_kilometer_for_vehicle
from .exceptions import NoSolutionFoundException
from .routeplan_factory import route_plan_from_vehicle_trip_map
from .validation import check_trips_only_has_single_date

# Initialize logger.
log = logging.getLogger(__name__)

# The CP SAT solver works with integer linear expressions/terms, and thus, the length of trips is scaled to account for this.
SCALING_OF_TRIPS = 1000


class RoutingCp(RoutingAlgorithm):
    def optimize_single_day(
        self,
        trips: list[Trip],
        vehicles: list[Vehicle],
        employee_car: BaseVehicle,
        emission_cost_per_ton_co2: float = 1500,
        time_limit_in_seconds: int = 60,
    ) -> RoutePlan:
        """
        This is an exact SAT algorithm that assigns trips to vehicles based on a weight of the variable cost per kilometer and the CO2 emission. time_limit_in_seconds specifies for how long, the algorithm is allowed to be run.
        :param trips: List of trips in the route plan.
        :param vehicles: List of vehicles in the route plan.
        :param employee_car: Employee car a trip can be assigned to.
        :param emission_cost_per_ton_co2: CO2 emission cost per ton for the entire route plan.
        :param time_limit_in_seconds: Time limit for the running time of the algorithm.
        :return Routing plan created after optimization has been performed.
        """

        log.info("Running exact SAT formulation of single day optimization.")

        # Check the trips. If any issues occur, throw an exception.
        check_trips_only_has_single_date(trips)

        log.debug("Checked trips input for being single day.")

        # Create the mip solver with the SCIP backend.
        model = cp_model.CpModel()

        # Holds all the variables related to the normal vehicles.
        vehicles_var = {}

        for vehicle in vehicles:
            for trip in trips:
                # Make a binary variable that is 1 if the trip is assigned to the vehicle, otherwise 0.
                vehicles_var[(vehicle, trip)] = model.NewBoolVar(
                    f"vehicle_{vehicle.id}_trip_{trip.id}."
                )

        log.debug("Created variables for normal vehicles.")

        # Holds all the variables related to the employee car as this has some special logical later on.
        employee_car_var = {}

        for trip in trips:
            # Make a binary variable that is 1 if the trip is assigned to the employee car, 0 otherwise.
            employee_car_var[trip] = model.NewBoolVar(f"employee_car_trip_{trip.id}.")

        log.debug("Created variables for employee car.")

        # Create constraints to ensure that a vehicle cannot serve overlapping trips.
        for first_trip in trips:
            for second_trip in trips:
                # Do nothing if the trips are the same.
                if first_trip == second_trip:
                    continue

                # Check if the two trips overlap in time.
                # See https://stackoverflow.com/questions/325933/determine-whether-two-date-ranges-overlap
                if max(first_trip.start_time, second_trip.start_time) < min(
                    first_trip.end_time, second_trip.end_time
                ):
                    # Overlap and add a constraint for each normal vehicle.
                    for vehicle in vehicles:
                        model.Add(
                            vehicles_var.get((vehicle, first_trip))
                            + vehicles_var.get((vehicle, second_trip))
                            <= 1
                        )

        log.debug(
            "Created constraints to ensure that overlapping trips are not assigned to the same vehicle."
        )

        # Create constraints to ensure all trips are assigned to a vehicle (normal or employee car).
        for trip in trips:
            model.Add(
                sum(vehicles_var.get((vehicle, trip)) for vehicle in vehicles)
                + employee_car_var.get(trip)
                == 1
            )

        log.debug(
            "Created constraints to ensure each trip is assigned to a vehicle or employee car."
        )

        # Create a constraint on the maximum range and the maximum time  for a vehicle.
        for vehicle in vehicles:
            # Constraint for maximum range.
            model.Add(
                sum(
                    int(SCALING_OF_TRIPS * trip.length_in_kilometers)
                    * vehicles_var.get((vehicle, trip))
                    for trip in trips
                )
                <= int(SCALING_OF_TRIPS * vehicle.range_in_kilometers)
            )
            # Constraint for maximum uptime.
            model.Add(
                sum(
                    trip.get_trip_length_in_minutes()
                    * vehicles_var.get((vehicle, trip))
                    for trip in trips
                )
                <= vehicle.maximum_driving_in_minutes
            )

        log.debug("Created range constraints for each vehicle.")

        # All objective terms.
        objective_terms = []

        for trip in trips:
            # Add term for employee car.
            objective_terms.append(
                trip.length_in_kilometers
                * (
                    employee_car.variable_cost_per_kilometer
                    + calculate_co2_emission_cost_per_kilometer_for_vehicle(
                        employee_car, emission_cost_per_ton_co2
                    )
                )
                * employee_car_var.get(trip)
            )

            # Add terms for all the other vehicles.
            for vehicle in vehicles:
                objective_terms.append(
                    trip.length_in_kilometers
                    * (
                        vehicle.variable_cost_per_kilometer
                        + calculate_co2_emission_cost_per_kilometer_for_vehicle(
                            vehicle, emission_cost_per_ton_co2
                        )
                    )
                    * vehicles_var.get((vehicle, trip))
                )

        model.Minimize(sum(objective_terms))

        log.debug("Created objective function.")

        log.info("About to solve the optimization problem using the SAT solver.")

        solver = cp_model.CpSolver()

        solver.parameters.max_time_in_seconds = time_limit_in_seconds

        status = solver.Solve(model)

        log.info(f"Optimization terminated with status {status}.")

        # Store for each vehicle and employee car the trips assigned to it. Initialized with an empty list.
        assignments: {Vehicle: list} = {vehicle: [] for vehicle in vehicles}
        assignments.update({employee_car: []})

        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:

            log.info(f"Objective value was {solver.ObjectiveValue()}")

            for trip in trips:

                # Check if trip was assigned to the employee car.
                if solver.Value(employee_car_var.get(trip)):
                    log.debug(
                        f"Trip with id {trip.id} was assigned to the employee car."
                    )
                    assignments.get(employee_car).append(trip)
                    # Go to next trip.
                    continue

                for vehicle in vehicles:
                    if solver.Value(vehicles_var.get((vehicle, trip))):
                        log.debug(
                            f"Trip with id {trip.id} was assigned to vehicle with id {vehicle.id}."
                        )
                        assignments.get(vehicle).append(trip)
                        # Go to next trip.
                        break

        else:
            log.info("The problem does not seem to have an optimal solution.")
            raise NoSolutionFoundException()

        log.debug("Solution extracted from SAT and returned.")

        return route_plan_from_vehicle_trip_map(assignments)
