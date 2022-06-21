import datetime
import json
import os
import pickle
import time
import urllib.parse

import pandas as pd
import requests
from sqlalchemy import and_
from sqlalchemy.orm.query import Query

try:
    from fleetmanager.model.roundtripgenerator import trip_aggregator
    from fleetmanager.data_access import (
        AllowedStarts,
        Cars,
        RoundTrips,
        FuelTypes,
        VehicleTypes,
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
        FuelTypes,
        VehicleTypes,
        Trips,
        engine_creator,
        session_factory,
    )

path_default_cars = "_default_cars.pkl"
path_default_allowed_starts = "_default_starts.pkl"

if os.path.exists(path_default_cars):
    cleaned_cars = pickle.load(open(path_default_cars, "rb"))
else:
    cleaned_cars = pd.DataFrame()
if os.path.exists(path_default_allowed_starts):
    cleaned_starts = pickle.load(open(path_default_allowed_starts, "rb"))
else:
    cleaned_starts = pd.DataFrame()


def quantize_months(start_date, end_date, days=28):
    """
    Getting tuples of dates from start_date to end_date split by days length.

    Parameters
    ----------
    start_date  :   datetime, the date to start the quantisation
    end_date    :   datetime, the date to end the quantisation
    days    :   time between output (start, end)

    Returns
    -------
    list of date tuples
    """
    month_tuples = []
    date_index = start_date
    stop = False
    while True:
        next_month = date_index + datetime.timedelta(days=days)
        if next_month > end_date:
            stop = True
            next_month = end_date
        if (next_month - date_index).days == days or stop:
            month_tuples.append((date_index.isoformat(), next_month.isoformat()))
        date_index = next_month
        if stop:
            break
    return month_tuples


def run_request(uri, params):
    """
    Wrapper to retry request if it fails
    """
    while True:
        response = requests.get(uri, params=params)
        if response.status_code != 429:
            break
        print("Too many requests retrying...")
        time.sleep(3)  # Handle too many requests
    return response


def get_latlon_address(address):
    """
    Fallback function if no gps coordination is associated with the address
    """
    osm_url = (
        "https://nominatim.openstreetmap.org/search/"
        + urllib.parse.quote(address)
        + "?format=json"
    )
    response = requests.get(osm_url).json()
    if len(response) == 0:
        return [None, None]
    return [float(response[0]["lat"]), float(response[0]["lon"])]


def get_or_create(Session, model, parameters):
    """
    Search for an object in the db, create it if it doesn't exist
    return on both scenarios
    """
    with Session.begin() as session:
        instance = session.query(model).filter_by(id=parameters["id"]).first()
        if instance:
            session.expunge_all()
    if instance:
        return instance
    else:
        instance = model(**parameters)
        with Session.begin() as session:
            session.add(instance)
            session.commit()
        return instance


def update_car(vehicle, saved_car):
    """returns true if saved car values are not equal to the new vehicle input"""
    for key, value in vehicle.items():
        if pd.isna(value) and pd.isna(saved_car[key]):
            continue
        if value != saved_car[key]:
            return True
    return False


blacklist_id = []  # placeholder for location that you don't want loaded
id_to_latlon = {}  # placeholder for places that don't have an associated lat/lon
fuel_to_type = {
    # benzin til fossilbil
    1: 4,
    # diesel til fossilbil
    2: 4,
    # el til elbil
    3: 3,
}

default_types = {
    "Zoe": {"fuel": 3, "type": 3},
    "Kona": {"fuel": 3, "type": 3},
    "E-NV200": {"fuel": 3, "type": 3},
    "Berlingo": {"fuel": 1, "type": 4},
    "Transit": {"fuel": 2, "type": 4},
    "Caddy": {"fuel": 1, "type": 4},
    "Yaris": {"fuel": 1, "type": 4},
    "Bipper": {"fuel": 2, "type": 4},
    "Jumper": {"fuel": 2, "type": 4},
    "Partner": {"fuel": 2, "type": 4},
    "Kangoo": {"fuel": 3, "type": 3},
}

if __name__ == "__main__":
    key = os.environ["API_KEY"]
    url = "https://app.ecofleet.com/seeme/"

    now = datetime.datetime.now()

    engine = engine_creator(
        db_name=os.environ["DB_NAME"],
        db_password=os.environ["DB_PASSWORD"],
        db_user=os.environ["DB_USER"],
        db_url=os.environ["DB_URL"],  # "host.docker.internal:3306",#"localhost",
        db_server="mysql",
    )
    Session = session_factory(engine)

    params = {"key": key, "json": ""}
    # default values
    sacred_starts = []

    for clean_start in cleaned_starts.to_dict("records"):
        sacred_starts.append(clean_start["id"])
        clean_start = {
            key: None if pd.isna(value) else value
            for key, value in clean_start.items()
            if key in AllowedStarts.__dict__.keys()
        }
        get_or_create(Session, AllowedStarts, clean_start)

    # AllowedStarts start
    places_response = requests.get(url + "Api/Places/get", params=params)
    allowed_starts = []
    current_starts = pd.read_sql(Query(AllowedStarts).statement, engine)
    for place in json.loads(places_response.content)["response"]:
        id_ = place["id"]
        if (
            id_ in blacklist_id
            or id_ in current_starts.id.values
            or id_ in sacred_starts
        ):
            # don't add if the place already exist
            continue
        lat, lon = get_latlon_address(place["name"])
        if lat is None and id_ not in id_to_latlon:
            print(f"Failed getting lat/lon for place(id={id_}, name={place['name']})")
            continue
        if lat is None:
            lat, lon = id_to_latlon[place["id"]]
        allowed_starts.append(
            AllowedStarts(
                id=int(place["id"]), address=place["name"], latitude=lat, longitude=lon
            )
        )
    with Session.begin() as sess:
        sess.add_all(allowed_starts)
    # AllowedStarts end

    # default values
    sacred = []
    for clean_car in cleaned_cars.to_dict("records"):
        sacred.append(clean_car["id"])
        clean_car = {
            key: None if pd.isna(value) else value
            for key, value in clean_car.items()
            if key in Cars.__dict__.keys()
        }
        get_or_create(Session, Cars, clean_car)

    # Cars start
    # getting vehicle settings for setting type ids on the cars
    starts = pd.read_sql(Query(AllowedStarts).statement, engine)
    fuel_settings = pd.read_sql(Query(FuelTypes).statement, engine)
    vehicle_settings = pd.read_sql(Query(VehicleTypes).statement, engine)
    vehicles_response = requests.get(url + "Api/Vehicles/get", params=params)
    cars = json.loads(vehicles_response.content)["response"]
    # get currently saved cars to update if changes and save new ones
    current_cars = pd.read_sql(Query(Cars).statement, engine)
    update_cars = []
    ups = []
    for car in cars:
        id_ = car["id"]

        if (
            pd.isna(car["booking"]["homeLocation"])
            or car["booking"]["homeLocation"] not in starts.id.values
        ):
            print(
                f"Car {id_} did not have any homeLocation or saved location: {car['booking']['homeLocation']}"
            )
            continue

        if id_ in sacred:
            # todo temporary value to persist the default values
            continue

        plate = car["plate"]
        if plate is not None and len(plate) > 7:
            plate = plate[:7]

        if all(
            [plate is None, car["info"]["make"] is None, car["info"]["model"] is None]
        ):
            continue

        fuel = None
        vehicle_type = None
        if car["info"]["fuelType"]:
            fuel = car["info"]["fuelType"]
            if fuel in fuel_settings.name.values:
                fuel = int(
                    fuel_settings[fuel_settings.name == fuel].refers_to.values[0]
                )
            else:
                fuel = None

        if car["info"]["vehicleType"]:
            # check the refers to
            vehicle_type = car["info"]["vehicleType"]
            if vehicle_type in vehicle_settings.name.values:
                vehicle_type = int(
                    vehicle_settings[
                        vehicle_settings.name == vehicle_type
                    ].refers_to.values[0]
                )
            else:
                vehicle_type = None
        if vehicle_type is None and fuel:
            # best guess
            vehicle_type = fuel_to_type[fuel]

        model = car["info"]["model"]
        if vehicle_type is None and fuel is None and model in default_types.keys():
            fuel = default_types[model]["fuel"]
            vehicle_type = default_types[model]["type"]

        location = (
            None
            if car["booking"]["homeLocation"] is None
            else int(car["booking"]["homeLocation"])
        )
        car_details = dict(
            id=car["id"],
            plate=plate,
            make=car["info"]["make"],
            model=car["info"]["model"],
            type=vehicle_type,
            fuel=fuel,
            # todo implement "auto fill" if the below metrics doesn't exist and similar make model exist
            wltp_fossil=None,  # todo update when we receive confirmation
            wltp_el=None,  # todo update when we receive confirmation
            co2_pr_km=None,  # todo update when we receive confirmation
            range=None,  # todo update when we receive confirmation
            location=location,
        )

        update_existing_car = False
        if id_ in current_cars.id.values:
            if not update_car(
                car_details, current_cars[current_cars.id == id_].iloc[0]
            ):
                continue
            else:
                current_car = get_or_create(Session, Cars, {"id": id_})
                update_existing_car = True

        if update_existing_car:
            for key, value in car_details.items():
                if key == "id":
                    continue
                setattr(current_car, key, value)
            update_cars.append(current_car)
        else:
            update_cars.append(Cars(**car_details))
    if update_cars:
        with Session.begin() as sess:
            sess.add_all(update_cars)
    # Cars end

    # Trips start
    start_date = None
    all_cars = pd.read_sql(Query(Cars).statement, engine)
    start_locations = pd.read_sql(Query(AllowedStarts).statement, engine)
    address2id = {a.address: a.id for a in start_locations.itertuples()}

    for car in all_cars.itertuples():
        lat, lon, address_id = None, None, None
        if not pd.isna(car.location) and int(car.location) in start_locations.id.values:
            lat, lon, address_id = start_locations[start_locations.id == car.location][
                ["latitude", "longitude", "id"]
            ].values[0]
        else:
            continue

        car_trips = []  # for holding new unseen trips
        untripped = []  # for holding saved trips that is not part of a roundtrip
        saved_id = []
        tripped = 0
        start_date = datetime.datetime(year=2022, month=1, day=20)
        if car.id in current_cars.id.values:
            last_roundtrip = pd.read_sql(
                Query(RoundTrips.end_time)
                .filter(RoundTrips.car_id == car.id)
                .order_by(RoundTrips.start_time.desc())
                .limit(1)
                .statement,
                engine,
            ).end_time
            # pull in trips from last untripped trips
            if len(last_roundtrip):
                pull_date = last_roundtrip[0]
            else:
                pull_date = start_date
            untripped = pd.read_sql(
                Query(Trips)
                .filter(and_(Trips.car_id == car.id, Trips.start_time > pull_date))
                .order_by(Trips.start_time.asc())
                .statement,
                engine,
            )
            if len(untripped):
                start_date = untripped.iloc[-1].end_time
                untripped = untripped
                saved_id = untripped.id.values
            else:
                start_date = pull_date
                untripped = pd.DataFrame()
        month_pairs = quantize_months(start_date, now)

        for k, (start_month, end_month) in enumerate(month_pairs):
            params = {
                "key": key,
                "objectId": car.id,
                "begTimestamp": start_month,
                "endTimestamp": end_month,
                "json": "",
            }
            response_ = run_request(url + "Api/Vehicles/getTrips", params=params)
            response = json.loads(response_.content)["response"]
            if k == 0 and len(untripped) > 0:
                response = untripped.to_dict("records") + response
            for i, trip in enumerate(response):

                if trip["id"] not in saved_id:
                    start_location = (
                        None
                        if trip["startLocation"] not in address2id
                        else address2id[trip["startLocation"]]
                    )

                    car_trips.append(
                        dict(
                            id=trip["id"],
                            car_id=car.id,
                            distance=trip["distance"],
                            start_time=datetime.datetime.fromisoformat(
                                trip["startTimestamp"][:-5]
                            ),
                            end_time=datetime.datetime.fromisoformat(
                                trip["endTimestamp"][:-5]
                            ),
                            start_latitude=trip["startLatitude"],
                            start_longitude=trip["startLongitude"],
                            end_latitude=trip["endLatitude"],
                            end_longitude=trip["endLongitude"],
                            start_location=start_location,
                            driver_name=None,
                        )
                    )

        sanitised = []
        if car_trips:
            trimmed = {a["id"]: Trips(**a) for a in car_trips}
            with Session.begin() as sess:
                sess.add_all(list(trimmed.values()))
            if type(untripped) is list:
                untripped = pd.DataFrame()
            trips = pd.concat([untripped, pd.DataFrame(car_trips)])
            sanitised = trip_aggregator(car, trips, start_locations)

        if sanitised:
            with Session.begin() as sess:
                sess.add_all(
                    [
                        RoundTrips(
                            **{
                                key: None if pd.isna(value) else value
                                for key, value in route.items() if key in RoundTrips.__dict__.keys()
                            }
                        )
                        for route in sanitised
                    ]
                )

        if car_trips:
            print(
                f"Car: {car.id}, saved {len(trips)} new trips and {len(sanitised)} roundtrips. "
                f"Ratio {sum([len(a['ids']) for a in sanitised])/len(trips)}"
            )
