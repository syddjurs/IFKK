from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.orm.query import Query

from fleetmanager.data_access.db_engine import engine_creator
from fleetmanager.data_access.dbschema import RoundTrips


def generate_trips_simulation(
    pool_id: int, seed: int = None, padding: float = 1.2, dates: list = []
):
    """
    Generates two list of possible trips for specific pool

    Parameters
    ----------
    dates : list
        The selected start - and end date of simulation period
    pool_id : int
        The pool to simulate.
    seed : int
        The seed of the random function that samples trips.
    padding : float
        increases the amount of simulated trips by a percentage

    Returns
    -------
    simulated_day: list[Dict]
        A list of trips from an average simulated day.
    peak_day: list[Dict]
        The list of trips from the day with most trips.
    """
    if len(dates) == 0:
        dates = [datetime(year=2022, month=1, day=20), datetime.now()]
    engine = engine_creator()
    data = pd.read_sql(
        Query([RoundTrips.start_time, RoundTrips.end_time, RoundTrips.distance])
        .filter(
            RoundTrips.start_location_id == pool_id,
            RoundTrips.start_time >= dates[0],
            RoundTrips.end_time <= dates[-1],
        )
        .statement,
        engine,
        parse_dates={
            "start_time": {"format": "%Y/%m/%d"},
            "end_time": {"format": "%Y/%m/%d"},
        },
    )

    # Remove trips spanning multiple days
    mask = data.apply(lambda row: row["end_time"].day == row["start_time"].day, axis=1)
    data = data[mask]

    if data.size == 0:
        return [], []
    else:
        return (
            __simulate_avg_day(data, seed=seed, padding=padding),
            __extract_peak_day(data),
        )


def __minutes_since_midnight(timestamp):
    midnight = timestamp.replace(hour=0, minute=0, second=0)
    return (timestamp - midnight).seconds / 60


def __extract_peak_day(data):
    grouped_start_time = data.groupby([data["start_time"].dt.date])
    peak_day_date = grouped_start_time.size().sort_values(ascending=False).index[0]
    # Help
    day_mask = data.apply(
        lambda row: f"{row['start_time'].year}-{row['start_time'].month}-{row['start_time'].day}"
        == f"{peak_day_date.year}-{peak_day_date.month}-{peak_day_date.day}",
        axis=1,
    )

    peak_day_trips = data[day_mask].values.tolist()

    peak_day = []

    for i in range(len(peak_day_trips)):
        peak_day.append(
            {
                "id": i,
                "start_time": peak_day_trips[i][0].to_pydatetime(),
                "end_time": peak_day_trips[i][1].to_pydatetime(),
                "length_in_kilometers": peak_day_trips[i][2],
            }
        )

    return peak_day


def __simulate_avg_day(data, seed, padding):
    grouped_start_time = data.groupby([data["start_time"].dt.date])
    # Add 20% to compensate for missing trips in database
    avg_trips_pr_day = round(grouped_start_time.size().mean() * padding)

    km_pr_min = data.apply(
        lambda row: (((row["end_time"] - row["start_time"]).seconds / 60))
        / row["distance"],
        axis=1,
    ).mean()

    distances = data["distance"].tolist()
    start_times = data["start_time"].apply(
        lambda x: __minutes_since_midnight(x.to_pydatetime())
    )

    # Histogram bins: 1 bin pr 10 km and 1 bin pr. 15 min
    distance_bins = round((round(data["distance"].max() - data["distance"].min())) / 10)
    start_time_bins = round((start_times.max() - start_times.min()) / 15)
    if distance_bins == 0:
        simulated_day = []
        for k, trip in enumerate(data.itertuples()):
            simulated_day.append({
                "id": k,
                "start_time": trip.start_time,
                "end_time": trip.end_time,
                "length_in_kilometers": trip.distance
            })
        return simulated_day

    (
        hist,
        x_bins,
        y_bins,
    ) = np.histogram2d(distances, start_times, bins=(distance_bins, start_time_bins))
    x_bin_midpoints = (x_bins[:-1] + x_bins[1:]) / 2
    y_bin_midpoints = (y_bins[:-1] + y_bins[1:]) / 2

    cdf = np.cumsum(hist.flatten())
    cdf = cdf / cdf[-1]

    if seed != None:
        np.random.seed(seed)

    values = np.random.rand(avg_trips_pr_day)
    value_bins = np.searchsorted(cdf, values)

    x_idx, y_idx = np.unravel_index(
        value_bins, (len(x_bin_midpoints), len(y_bin_midpoints))
    )
    random_from_cdf = np.column_stack((x_bin_midpoints[x_idx], y_bin_midpoints[y_idx]))
    new_distances, new_start_times = random_from_cdf.T

    simulated_day = []

    for i in range(avg_trips_pr_day):
        start_time = datetime.now() + timedelta(days=365, minutes=new_start_times[i])
        end_time = start_time + timedelta(minutes=new_distances[i] * km_pr_min)
        simulated_day.append(
            {
                "id": i,
                "start_time": start_time,
                "end_time": end_time,
                "length_in_kilometers": new_distances[i],
            }
        )

    return simulated_day
