""" This file defines various helper functions used throughout the code."""

import copy

from .classes import BaseVehicle
from .cost_calculator import calculate_co2_emission_cost_per_kilometer_for_vehicle


def prioritize_vehicles_according_to_weighted_variable_costs(
    vehicles: list[BaseVehicle], emission_cost_per_ton_co2: float
) -> list[BaseVehicle]:
    """
    This function weights the variable costs and CO2 emission of the vehicles and returns them in a sorted list in ascending order.
    :param vehicles: Vehicles to be listed.
    :param emission_cost_per_ton_co2: The cost of CO2 emission in dkk (or at least same currency as the variable cost on the vehicles).
    :return: Prioritized list of vehicles with the cheapest vehicle in position 0, second cheapest in position 1, etc.
    """
    # Initialize list, that will be sorted, to not change input.
    vehicles_sorted = copy.deepcopy(vehicles)

    # Sort by the actual variable cost, which is the variable cost + CO2 emission equivalent cost.
    vehicles_sorted.sort(
        key=lambda x: (
            x.variable_cost_per_kilometer
            + calculate_co2_emission_cost_per_kilometer_for_vehicle(
                x, emission_cost_per_ton_co2
            )
        )
    )

    return vehicles_sorted
