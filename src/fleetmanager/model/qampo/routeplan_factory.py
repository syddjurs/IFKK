""" This file defines route plan factory to be used for classes.py."""

from .classes import Assignment, RoutePlan, Trip, Trips, Vehicle
from .cost_calculator import (
    calculate_co2_emission_ton_of_trips,
    calculate_variable_cost_of_trips,
)


def route_plan_from_vehicle_trip_map(
    vehicle_assignments: {Vehicle, list[Trip]}
) -> RoutePlan:
    """
    Creates a route plan from a dictionary of vehicle as key and list of trips as value.
    :param vehicle_assignments: Assignments for vehicle.
    :return: Route plan.
    """
    # Vehicle assignments.
    assignments = []

    # Total CO2 emission in tons.
    total_co2_emission_tons = 0.0

    # Total variable costs.
    total_variable_costs = 0.0

    employee_car_assignment = None

    for vehicle, trips in vehicle_assignments.items():
        co2_emission_of_assignment_tons = calculate_co2_emission_ton_of_trips(
            vehicle, trips
        )
        variable_cost_of_assignment = calculate_variable_cost_of_trips(vehicle, trips)
        assignment = Assignment(
            vehicle=vehicle,
            route=Trips(trips=trips),
            co2_emission_in_tons=co2_emission_of_assignment_tons,
            variable_cost=variable_cost_of_assignment,
        )
        total_variable_costs += variable_cost_of_assignment
        total_co2_emission_tons += co2_emission_of_assignment_tons

        # Check if it is a normal car or an employee car.
        if isinstance(vehicle, Vehicle):
            assignments.append(assignment)
        else:
            employee_car_assignment = assignment

    return RoutePlan(
        assignments=assignments,
        employee_car=employee_car_assignment,
        total_co2_emission_in_tons=total_co2_emission_tons,
        total_cost=total_variable_costs,
    )
