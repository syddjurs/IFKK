"""
This file defines the main functionality of the repository - a webservice that can be called to optimize routes a day/a fleet over a couple of days.
FastAPI is used for handling different HTTP requests, for instance to post an optimization.
"""

import logging
from typing import Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException

from .classes import AlgorithmParameters, AlgorithmType, Fleet, Trip
from .routing_cp import RoutingCp
from .routing_greedy import RoutingGreedy
from .routing_mip import RoutingMip

description = """
This service holds a number of endpoints used for optimization in the IFFK project. The idea is that the service can be applied to any project, and not to a specific one.
"""

app = FastAPI(
    title="Simulation tool optimization service",
    description=description,
    version="0.0.3",
)


@app.post("/optimize_single_day")
async def optimize_single_day(
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

        raise HTTPException(status_code=400, detail=details)


if __name__ == "__main__":

    with open("log-config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        logging.config.dictConfig(config)
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="debug")
