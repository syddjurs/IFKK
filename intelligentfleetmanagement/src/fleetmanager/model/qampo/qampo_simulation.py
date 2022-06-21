from typing import Optional

from .classes import AlgorithmParameters, AlgorithmType, Fleet, Trip
from .routing_cp import RoutingCp
from .routing_greedy import RoutingGreedy
from .routing_mip import RoutingMip


def optimize_single_day(
    fleet: Fleet,
    trips: list[Trip],
    algorithm_type: AlgorithmType,
    algorithm_parameters: Optional[AlgorithmParameters] = AlgorithmParameters(
        time_limit_in_seconds=60
    ),
):
    if algorithm_type is AlgorithmType.GREEDY:
        return RoutingGreedy().optimize_single_day(
            trips, fleet.vehicles, fleet.employee_car, fleet.emission_cost_per_ton_co2
        )
    elif algorithm_type is AlgorithmType.EXACT_MIP:
        return RoutingMip().optimize_single_day(
            trips,
            fleet.vehicles,
            fleet.employee_car,
            fleet.emission_cost_per_ton_co2,
            algorithm_parameters.time_limit_in_seconds,
        )
    elif algorithm_type is AlgorithmType.EXACT_CP:
        return RoutingCp().optimize_single_day(
            trips,
            fleet.vehicles,
            fleet.employee_car,
            fleet.emission_cost_per_ton_co2,
            algorithm_parameters.time_limit_in_seconds,
        )
    # Unsupported algorithm type.
    else:
        supported_algorithms = str.join(
            ", ", [algorithm.value for algorithm in AlgorithmType]
        )
        details = (
            f"Algorithm type is not supported. Supported types: {supported_algorithms}"
        )

        raise ValueError(details)
