#!/usr/bin/env python3
import os
import pickle
from datetime import datetime, timedelta

import click
import numpy as np
import pandas as pd
from parsers import DrivingBook, MileageLogPositions, Trackers
try:
    from fleetmanager.model.roundtripgenerator import trip_aggregator
    from fleetmanager.data_access import (
        AllowedStarts,
        Cars,
        RoundTrips,
        Trips,
        engine_creator,
        session_factory,
    )
except ImportError:
    from roundtripgenerator import trip_aggregator
    from data_access import (
        AllowedStarts,
        Cars,
        RoundTrips,
        Trips,
        engine_creator,
        session_factory,
    )

from soap_agent import SoapAgent
from sqlalchemy import and_, func
from sqlalchemy.orm.query import Query


@click.group()
@click.option("-d", "--db-name", envvar="DB_NAME", required=True)
@click.option("-p", "--password", envvar="DB_PASSWORD", required=True)
@click.option("-u", "--db-user", envvar="DB_USER", required=True)
@click.option("-l", "--db-url", envvar="DB_URL", required=True)
@click.option("-s", "--db-server", envvar="DB_SERVER", required=True)
@click.pass_context
def cli(ctx, db_name=None, password=None, db_user=None, db_url=None, db_server=None):
    """
    Preserves the context for the remaining functions
    Parameters
    ----------
    ctx
    db_name
    password
    db_user
    db_url
    db_server
    """
    ctx.ensure_object(dict)
    engine = engine_creator(
        db_name=db_name,
        db_password=password,
        db_user=db_user,
        db_url=db_url,
        db_server=db_server,
    )
    ctx.obj["engine"] = engine
    ctx.obj["Session"] = session_factory(engine)


@cli.command()
@click.pass_context
def set_roundtrips(ctx):
    """
    Function to handle the aggregation of trips to roundtrips
    """
    engine = ctx.obj["engine"]
    Session = ctx.obj["Session"]

    subq = latest_time_query(RoundTrips).subquery()
    trips = (
        pd.read_sql(
            Query(Trips)
            .join(
                subq,
                (Trips.car_id == subq.c.car_id)
                & (Trips.start_time > subq.c.lastest_time),
            )
            .union(
                Query(Trips).filter(
                    ~Query(Trips).filter(Trips.car_id == subq.c.car_id).exists()
                )
            )
            .statement,
            engine,
        )
        .dropna(subset=["trips_distance"])
        .sort_values("trips_start_time", ascending=True)
    )
    trips.rename(
        {key: "_".join(key.split("_")[1:]) for key in trips.columns},
        axis=1,
        inplace=True,
    )
    allowed_starts = pd.read_sql(Query(AllowedStarts).statement, engine)
    additional_big_parking_lot_starts = pd.DataFrame()
    if os.path.exists("default_additional_starts.pkl"):
        additional_big_parking_lot_starts = pickle.load(open("default_additional_starts.pkl", "rb"))
    allowed_starts = pd.concat([allowed_starts, additional_big_parking_lot_starts])
    cars = pd.read_sql(Query(Cars).statement, engine)
    for car in cars.itertuples():
        c_trips = trips[trips.car_id == car.id]
        sanitised = trip_aggregator(car, c_trips, allowed_starts=allowed_starts)

        if sanitised:
            with Session.begin() as sess:
                sess.add_all(
                    [
                        RoundTrips(
                            **{
                                key: None if pd.isna(value) else value
                                for key, value in route.items()
                                if key in RoundTrips.__dict__.keys()
                            }
                        )
                        for route in sanitised
                    ]
                )

        if len(c_trips) > 0:
            print(
                sum([len(a["ids"]) for a in sanitised]),
                len(c_trips),
                len(sanitised),
                len(sanitised),
            )
            print(
                f"Car: {car.id}, saved {len(c_trips)} new trips and {len(sanitised)} roundtrips. "
                f"Ratio {sum([len(a['ids']) for a in sanitised])/len(c_trips)}"
            )


@cli.command()
@click.pass_context
@click.option("-s", "--soap-key", envvar="SOAP_KEY", required=True)
def set_trackers(ctx, soap_key):
    """
    Loads the trackers, which in SkyHost terms is equivalent to a vehicle in their system.
    Due to the limitation on data served through their api, it heavily relies on supporting data.

    It is recommended to first run set_trackers to then manually update the Cars frame with associated data.

    AllowedStarts: is an essential part, which should be stored in a dataframe in pickle format, like:
                id, address, latitude, longitude
    Cars: relies on an associated AllowedStarts, i.e. its home location. If this pickle is not provided, only id of
            the vehicle will be stored with no relationship to a start.
    """
    Session = ctx.obj["Session"]
    agent = SoapAgent(soap_key)
    trackers = Trackers()
    if os.path.exists("default_starts.pkl"):
        default_starts = pickle.load(open("default_starts.pkl", "rb"))
        for start in default_starts.to_dict("records"):
            get_or_create(Session, AllowedStarts, start)

    if os.path.exists("default_cars.pkl"):
        default_cars = pickle.load(open("default_cars.pkl", "rb"))
    else:
        default_cars = pd.read_sql(Query(Cars).statement, ctx.obj["engine"])
    while (r := agent.execute_action("Trackers_GetAllTrackers")).status_code != 200:
        print("Retrying Trackers_GetAllTrackers")
    trackers.parse(r.text)
    for tracker in trackers.frame.itertuples():
        car = dict(
            id=tracker.ID
            # todo add the additional parameters when they're exposed by SkyHost
            # todo update if meta data on car changed
        )
        if tracker.Description in default_cars.plate.values:
            car_parameters = default_cars[
                default_cars.plate == tracker.Description
            ].to_dict("records")[0]
            del car_parameters["id"]
            car.update(car_parameters)
        get_or_create(Session, Cars, car)


@cli.command()
@click.pass_context
@click.option("-s", "--soap-key", envvar="SOAP_KEY", required=True)
def set_trips(ctx, soap_key):
    """
    Function for pulling the driving log (kørebog)
    """
    Session = ctx.obj["Session"]
    engine = ctx.obj["engine"]
    agent = SoapAgent(soap_key)
    start_time = {}
    start_pos = {}
    current_time = datetime.now()
    min_time = datetime(
        year=2022, month=2, day=24
    )  # 23/2 seems to be the date from which lat, lon is added
    with Session() as sess:
        for r in sess.execute(latest_time_query(Trips)):
            start_time[str(r.car_id)] = r.lastest_time
        for id, end in start_time.items():
            latest_trip = sess.execute(
                Query(Trips).filter(and_(Trips.car_id == id, Trips.end_time == end))
            ).first()
            latest_trip = latest_trip[0]
            sess.expunge(latest_trip)
            start_pos[id] = latest_trip

    cars = pd.read_sql(Query(Cars).statement, engine)
    for car in cars.itertuples():
        trips = []
        for k, (start_month, end_month) in enumerate(
            date_iter(
                start_time.get(str(car.id), min_time), current_time, week_period=52
            )
        ):
            print(start_month, end_month)
            dbook = driving_book(
                car.id,
                start_month.isoformat(),
                end_month.isoformat(),
                agent,
            )

            if not all(
                [
                    "StartPos_sLat" in dbook.columns,
                    "StartPos_sLon" in dbook.columns,
                    "StopPos_Timestamp" in dbook.columns,
                ]
            ):
                continue

            trimmed = dbook[
                (dbook.StartPos_sLat.notnull())
                & (dbook.StartPos_sLon.notnull())
                & (dbook.StopPos_Timestamp.notnull())
            ]

            if len(trimmed) != len(dbook):
                print(
                    f"Car {car.id} did not have lat, lon or StopPost_Timestamp in all "
                    f"logs between {start_month} - {end_month}"
                )
                break
            for trip in dbook.itertuples():
                trips.append(
                    dict(
                        id=trip.ID,
                        car_id=car.id,
                        distance=int(trip.Meters) / 1000,
                        start_time=datetime.strptime(
                            trip.StartPos_Timestamp, "%Y-%m-%dT%H:%M:%S"
                        ),
                        end_time=datetime.strptime(
                            trip.StopPos_Timestamp, "%Y-%m-%dT%H:%M:%S"
                        ),
                        start_latitude=trip.StartPos_sLat,
                        start_longitude=trip.StartPos_sLon,
                        end_latitude=None,
                        end_longitude=None,
                        # todo add start location when we get it
                    )
                )

        if len(trips) == 0:
            continue
        trips_frame = pd.DataFrame(trips)
        trips_frame["end_latitude"] = trips_frame["start_latitude"].tolist()[1:] + [
            None
        ]
        trips_frame["end_longitude"] = trips_frame["start_longitude"].tolist()[1:] + [
            None
        ]
        with Session.begin() as sess:
            add_these = [Trips(**trip) for trip in trips_frame.to_dict("records")]
            if str(car.id) in start_pos:
                update_trip = start_pos[str(car.id)]
                end_lat, end_lon = trips_frame.sort_values(
                    ["end_time"], ascending=True
                ).iloc[0][["start_latitude", "start_longitude"]]
                update_trip.end_latitude = end_lat
                update_trip.end_longitude = end_lon
                add_these.append(update_trip)

            sess.add_all(add_these)


def latest_time_query(Table):
    """
    Convenience function to get the last entry for cars
    """
    q = Query([Table.car_id, func.max(Table.end_time).label("lastest_time")]).group_by(
        Table.car_id
    )
    return q


def get_or_create(Session, model, parameters):
    """
    Convenience function to see if an entry already exists. Update it if it exists or else create it.
    returns the entry.
    """
    with Session.begin() as session:
        instance = session.query(model).filter_by(id=parameters["id"]).first()
        if instance:
            session.expunge_all()
    if instance:
        return instance
    else:
        instance = model(
            **{
                key: None if pd.isna(value) else value
                for key, value in parameters.items()
                if key in model.__dict__.keys()
            }
        )
        with Session.begin() as session:
            session.add(instance)
            session.commit()
        return instance


def get_coords(trip_id, agent):
    """
    Additional function if one needs to get logs further back than february 2022, since the MilageLog does not
    contain GPS coordinates.
    """
    while (
        r := agent.execute_action(
            "Trackers_GetMilagePositions", params={"MilageLogID": trip_id}
        )
    ).status_code != 200:
        print(
            "Retrying Trackers_GetMilagePositions with MilageLogId: {}".format(trip_id),
        )
    click.echo(r.text)
    log_pos = MileageLogPositions()
    log_pos.parse(r.text)
    return log_pos.frame.sort_values(by="Timestamp", ascending=True)


def driving_book(tracker_id, start_time, end_time, agent):
    """
    Function for pulling the MilageLog, which make up the entries in the Trips table
    """
    while (
        r := agent.execute_action(
            "Trackers_GetMilageLog",
            params={
                "TrackerID": tracker_id,
                "Begin": start_time,
                "End": end_time,
            },
        )
    ).status_code != 200:
        print(
            "Retrying Trackers_GetMilageLog with TrackerID: {} Begin: {} End: {}".format(
                tracker_id, start_time, end_time
            )
        )
    book = DrivingBook()
    book.parse(r.text)
    print(
        "Found {} trips for tracker with id {}".format(book.frame.shape[0], tracker_id)
    )
    if book.frame.shape[0] == 0:
        return book.frame
    else:
        return book.frame.sort_values(by="StartPos_Timestamp", ascending=True).replace(
            [np.nan], [None]
        )


def date_iter(start_date, end_date, week_period=24):
    """
    Function for iterating over a date period

    Parameters
    ----------
    start_date
    end_date
    week_period :   the period between the returned start and end date

    Returns
    -------
    start_date, end_date with week_period in between
    """
    delta = timedelta(weeks=week_period)
    last_start = start_date
    stopped = False
    while stopped is False:
        start_date += delta
        if start_date > end_date:
            start_date = end_date
            stopped = True
        yield last_start, start_date
        last_start = start_date


if __name__ == "__main__":
    cli()
