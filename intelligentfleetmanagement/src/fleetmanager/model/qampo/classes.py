""" This file defines the different classes used in the solution."""

import datetime
import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class Trip(BaseModel):
    """A trip consist of a start time, an end time and the length of the trip measured in kilometers.\n
    The starting location and ending location are assumed to be identical for this application."""

    id: int
    """A unique id for the trip."""

    start_time: datetime.datetime
    """The time for which the trip begins."""

    end_time: datetime.datetime
    """The time for which the trip ends."""

    length_in_kilometers: float
    """The length of the trip measured in kilometers."""

    # Makes it possible to make the Trip class hashable and used as key in map.
    class Config:
        allow_mutation = False

    def get_trip_length_in_minutes(self) -> int:
        """Get the trip duration in minutes (rounded up) of the trip
        @:return duration in number of minutes (rounded up)
        """
        return math.ceil((self.end_time - self.start_time).total_seconds() / 60.0)

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class Trips(BaseModel):
    """A list of trips."""

    trips: list[Trip]
    """A list of trips."""


class BaseVehicle(BaseModel):
    """A base vehicle consists of a variable cost of kilometers and an amount of CO2 emission per kilometer measured in grams."""

    variable_cost_per_kilometer: float
    """The variable cost per kilometer."""

    co2_emission_gram_per_kilometer: float
    """The CO2 emission per kilometer measured in grams."""

    # Makes it possible to make the Trip class hashable and used as key in map.
    class Config:
        allow_mutation = False

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class Vehicle(BaseVehicle):
    """A vehicle consists of a range of kilometers, a variable cost of kilometers, an amount of CO2 emission per kilometer measured in grams, and a name."""

    id: int
    """A unique id for the vehicle."""

    range_in_kilometers: float
    """The range in kilometers."""

    maximum_driving_in_minutes: int
    """ The number of minutes the vehicle can drive. Used to control e.g. electric cars recharge time in lieu of a proper battery management implementation e.g. an uptime of 16*60 minutes could model an 8 hour recharging time."""

    name: str = ""
    """The name of the car, e.g. Toyota Yaris No. 1. Not used in the algorithm, but for easier debugging and information."""

    # Makes it possible to make the Trip class hashable and used as key in map.
    class Config:
        allow_mutation = False

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class Fleet(BaseModel):
    """A tradeoff between C02 and money for a list of vehicles.\n
    Official documents say, somewhere between 1500 and 5000 dkk/ton is an optional list for possible parameters for scenarios, otherwise, default values are set."""

    vehicles: list[Vehicle]
    """A list of vehicles."""

    employee_car: BaseVehicle
    """An employee car."""

    emission_cost_per_ton_co2: Optional[float]
    """The CO2 emission cost per ton. Default value will be 1000 dkk."""


class Assignment(BaseModel):
    """An assignment is trips assigned to a specific vehicle."""

    vehicle: BaseVehicle
    """The vehicle, a route is assigned to."""

    route: Trips
    """Contains a list of trips. A new trip can start immediately, after the previous trip ends, thus, ending at 08:00 means, the next step can start at 08:00."""

    variable_cost: float
    """The variable cost for the route."""

    co2_emission_in_tons: float
    """The CO2 emission of the route measured in tons."""


class RoutePlan(BaseModel):
    """A route plan consists of a list of assignments, total cost and total CO2 emission."""

    assignments: list[Assignment]
    """A list of assignments."""

    employee_car: Assignment
    """ The assignment for the employee car(s)"""

    total_cost: float
    """The total cost of the routes."""

    total_co2_emission_in_tons: float
    """The total amount of CO2 emission of the assignments measured in tons."""


class AlgorithmType(Enum):
    """What type of algorithm to be used."""

    GREEDY = "greedy"
    """Greedy algorithm."""

    EXACT_MIP = "exact_mip"
    """Exact mixed-integer programming algorithm."""

    EXACT_CP = "exact_cp"
    """Exact constraint programming algorithm."""


class AlgorithmParameters(BaseModel):
    time_limit_in_seconds: Optional[int]
    """Defines for how long, an algorithm is allowed to be run."""


class RoutingAlgorithm(ABC):
    """An ABC for doing a single day routing assignment."""

    @abstractmethod
    def optimize_single_day(
        self,
        trips: list[Trip],
        vehicles: list[Vehicle],
        employee_car: BaseVehicle,
        emission_cost_per_ton_co2: float = 1500,
        time_limit_in_seconds: int = 60,
    ) -> RoutePlan:
        pass
