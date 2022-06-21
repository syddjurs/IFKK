""" This file defines functions for calculating the CO2 emission and costs for a route plan."""
from .classes import BaseVehicle, Trip


def calculate_total_length_of_trips(trips: list[Trip]) -> float:
    """Calculates the length of all trips combined."""
    return sum(trip.length_in_kilometers for trip in trips)


def calculate_variable_cost_of_trips(vehicle: BaseVehicle, trips: list[Trip]) -> float:
    """Calculates the cost of all trips combined."""
    return vehicle.variable_cost_per_kilometer * calculate_total_length_of_trips(trips)


def calculate_co2_emission_gram_of_trips(
    vehicle: BaseVehicle, trips: list[Trip]
) -> float:
    """Calculates the CO2 emission (grams) of all trips combined."""
    return vehicle.co2_emission_gram_per_kilometer * calculate_total_length_of_trips(
        trips
    )


def calculate_co2_emission_ton_of_trips(
    vehicle: BaseVehicle, trips: list[Trip]
) -> float:
    """Calculates the CO2 emission (tons) of all trips combined."""
    return calculate_co2_emission_gram_of_trips(vehicle, trips) / 1000000.00


def calculate_co2_emission_cost_per_kilometer_for_vehicle(
    vehicle: BaseVehicle, emission_cost_per_ton_co2: float
) -> float:
    """Calculates the CO2 emission cost per kilometer for a given vehicle."""
    return vehicle.co2_emission_gram_per_kilometer * emission_cost_per_ton_co2 / 1000000
