import datetime
import math
import operator

import numpy as np
import pandas as pd


def calc_distance(coord1, coord2):
    """
    Simple distance function to measure the distance in km from two coordinates (lat, long) (lat, long)
    Parameters
    ----------
    coord1 : (latitude, longitude)
    coord2 : (latitude, longitude)

    Returns
    -------
    distance in km
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371
    phi1 = lat1 * math.pi / 180
    phi2 = lat2 * math.pi / 180
    delta_phi = (lat2 - lat1) * math.pi / 180
    delta_lambda = (lon2 - lon1) * math.pi / 180

    a = math.sin(delta_phi / 2) * math.sin(delta_phi / 2) + math.cos(phi1) * math.cos(
        phi2
    ) * math.sin(delta_lambda / 2) * math.sin(delta_lambda / 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def point_to_starts(coordinate, starts):
    """
    Function used to measure the distance between a selected start to a list of starts
    iterates over the starts list and returns the distance and coordinates to the starting location
    to which there's the shortest distance.

    Parameters
    ----------
    coordinate : (lat, lon)
    starts : list of (lat, lon) - [(lat, lon), (lat, lon), (lat, lon) ...]

    Returns
    -------
    (min_distance, best_start) - distance to closest start, (lat, lon) to closest start
    """
    min_distance = math.inf
    best_start = None
    for start in starts:
        distance = calc_distance(start, coordinate)
        if distance < min_distance:
            min_distance = distance
            best_start = start
    return min_distance, best_start


def get_car_starts(trips, allowed_starts_):
    """
    Function to retrieve allowed starts for the cars. Iterates over all trips for a specific car
    and finds the start to which it's closest. If the distance to an allowed start is less than .2 km
    the start is accepted. The 3 most frequent starts for each car is returned for use in the later
    process when the roundtrips are defined

    Parameters
    ----------
    trips : trips from the table expected in pandas format. At least "car_id", "start_latitude", "start_longitude"
    allowed_starts : list of cordinates (lat, lon) that are accepted start locations

    Returns
    -------
    dictionary for each car id that contain the 3 most frequent starts for the car
    {car_id: [(lat,lon), (lat,lon), (lat,lon)], car_id: [(lat,lon), (lat,lon), (lat,lon)] ...}

    """
    cars = trips.car_id.unique()
    car_dict = {car: {} for car in cars}
    allowed_starts_coordinates = [
        (a.latitude, a.longitude) for a in allowed_starts_.itertuples()
    ]
    for car in cars:
        log_points = trips[trips.car_id == car]
        car_starts = [
            (a, b)
            for a, b in zip(log_points.start_latitude, log_points.start_longitude)
            if all([type(a) is not None, type(b) is not None])
        ]
        closest = [
            point_to_starts(start, allowed_starts_coordinates) for start in car_starts
        ]
        for distance, (points) in closest:
            if distance < 0.2:
                if points not in car_dict[car]:
                    car_dict[car][points] = 0
                car_dict[car][points] += 1

    allowed_car_starts = {
        car: [
            point
            for point, frequency in sorted(
                car_dict[car].items(), key=operator.itemgetter(1), reverse=True
            )[:3]
        ]
        for car in cars
    }
    return allowed_car_starts


def post_routing_sanitation(routes, trips, starts_):
    """
    Function used post routing to do sanitation - especially useful to scrutinise the trips that doesn't "make sense"
    from a distance or time perspective.
    The time (7 days) and distance (200 km) criteria defines which routes will be sought to be re-defined.
    For the selected routes it selects the trips points from the start - and end time, new starting locations are
    defined - isolated to those present as opposed to all log points for the car. The trips log points and new
    locations are sent to the trip extractor to define possible new routes.

    Parameters
    ----------
    routes : the defined routes from trip_extractor
    trips : the trips frame
    starts_ : frame holding the allowed starts from the allowed_starts table

    Returns
    -------
    routes that were originally accepted and possible sanitized routes
    """
    distance_criteria = 200
    time_criteria = 1

    confirmed = []
    check = []
    sanitized = []

    for route in routes:
        if (
            route["end_time"].date() - route["start_time"].date()
        ).days >= time_criteria or route["distance"] > distance_criteria:
            check.append(route)
        else:
            confirmed.append(route)

    for route in check:
        # check which starts it's most in relation with
        # select one start based on frequency
        car = route["car_id"]
        trip_points = trips[
            (trips.car_id == route["car_id"])
            & (trips.start_timestamp >= route["start_time"])
            & (trips.end_timestamp <= route["end_time"])
        ]
        new_starts = {car: get_car_starts(trip_points, starts_)[car][:]}
        new_routes, _ = trip_extractor(
            trip_points,
            allowed_car_starts=new_starts,
            start_frame=starts_,
        )
        sanitized += new_routes

    return confirmed + sanitized


def trip_extractor(
    frame,
    allowed_car_starts=None,
    start_frame=None,
    hour_threshold=16.9,
    definition_of_home=0.2,
    min_travel_distance=0.5,
    recurs=0,
    enforce_location=None,
):
    """
    Major roundtrip generator function. Takes the trips pulled and the defined allowed starts
    for each car. Assumes that a route will start and end at the same location (+-definition of home), exists of
    at least 2 log points, travelled distance is more than 500 meters and driver is the same
    (except if driver defined moves from nan to driver_name or driver_name to nan once)

    Parameters
    ----------
    frame : trips frame from trips table
    allowed_car_starts : dictionary of allowed starts for each car_id present in trips frame
    start_frame: frame holding the allowed starts from the allowed_starts table
    hour_threshold  :   int, standard for limiting the allowed length of a trip.
    definition_of_home  :   int, allowed distance to home. If a log is definition_of_home close to an allowed start
                            at unstarted trip, the trip will begin. If a trip is underway, the trips ends if a log
                            is seen that is definition_of_home close to the defined allowed start
    min_travel_distance :   int, the minimum travel distance for accepting the trip as a valid roundtrip
    recurs  :   int, should not be changed. Internal value for constraining the function to only recurs once to parse
                unqualified roundtrips in to qualified roundtrips. E.g. There could be vehicles that find home after
                days because it has been lend out to another location, then the allowed start has been "wrongly" selected
                and should be defined to the location to which the vehicle has been lend in order to record the trips
                driven in the other location.
    enforce_location    :   int, defining the location. In accordance with the aforementioned, this value will overwrite
                            the found location and always allocate the trip to the location to where the vehicle belong

    Returns
    -------
    list of accepted routes in dictionary format including following attributes: start_time, end_time, start_point,
        end_point, car_id, distance, gps_points, driver, ids

    """
    time_sorted_frame = frame.sort_values(["start_time"]).reset_index().iloc[:, 1:]
    time_sorted_frame["start_timestamp"] = time_sorted_frame.start_time.apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        if type(x) is str
        else x
    )
    time_sorted_frame["end_timestamp"] = time_sorted_frame.end_time.apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        if type(x) is str
        else x
    )

    all_routes = []  # list for qualified routes
    for car_id in time_sorted_frame.car_id.unique():
        current_route = []  # list for holding log points for current running route
        trip_end = False
        trip_frame = time_sorted_frame[time_sorted_frame.car_id == car_id]
        trip_frame_length = len(trip_frame)
        driver = trip_frame.iloc[0]["driver_name"]
        if pd.isnull(driver):
            driver = "nan"
        for k, segment in enumerate(trip_frame.itertuples()):
            # to allow nan driver log points to be allocated to route
            if pd.isnull(segment.driver_name):
                current_driver = "nan"
            else:
                current_driver = segment.driver_name

            # check that current driver is equal to the previous - except if one of them is a nan
            if (
                current_driver != driver and driver != "nan" and current_driver != "nan"
            ) or len(
                set(
                    [
                        "nan" if pd.isna(a.driver_name) else a.driver_name
                        for a in current_route
                    ]
                )
            ) == 3:
                # discard already logged points to the route because driver changed
                current_route = []

            if len(current_route) == 0:
                # if this is the first log of the trip ensure that the location is an accepted start
                distances_to_starts = [
                    calc_distance(
                        (segment.start_latitude, segment.start_longitude), (start)
                    )
                    for start in allowed_car_starts[car_id]
                ]
                if all([dist > definition_of_home for dist in distances_to_starts]):
                    continue
                test = 0
                driver = current_driver
                total_travelled = 0
                start_point = allowed_car_starts[car_id][
                    np.argsort(distances_to_starts)[0]
                ]
                start_time = segment.start_timestamp

                locat = start_frame[
                    (start_frame.latitude == start_point[0])
                    & (start_frame.longitude == start_point[1])
                ].id.values[0]
                allowed_stops = start_frame[start_frame.id == locat][
                    ["latitude", "longitude"]
                ].values
                multiple_stops = True if len(allowed_stops) > 1 else False

            current_route.append(segment)

            if k < trip_frame_length - 1:
                # due to "teleportation" where the current end coordinate is not the next start
                point = trip_frame.iloc[k + 1][
                    ["start_latitude", "start_longitude"]
                ].values
            else:
                point = (segment.end_latitude, segment.end_longitude)
            # we need the end location to determine whether it's parked close to home

            if multiple_stops:
                distances = [
                    calc_distance(point, (lat, lon)) for lat, lon in allowed_stops
                ]
                distance = distances[np.argsort(distances)[0]]
            else:
                distance = calc_distance(start_point, point)
            # sanity check (sometimes the gps distance is wrong)
            point_to_point = calc_distance(
                point, (segment.start_latitude, segment.start_longitude)
            )

            # gps points from skyhost is sometimes missing or (0, 0)
            if point_to_point > 5000:
                point_to_point = segment.distance
            try:
                total_travelled += (
                    point_to_point
                    if any(
                        [
                            point_to_point / segment.distance > 1.5,
                            0.05 > point_to_point / segment.distance,
                        ]
                    )
                    else segment.distance
                )
            except ZeroDivisionError:
                total_travelled += point_to_point
            # distance to the route start

            time_difference = segment.start_timestamp - start_time
            time_difference_seconds = (
                time_difference.days * 24 * 3600
            ) + time_difference.seconds
            # time to the last start checkpoint

            # if time exceeds 24 * 7 hours, trip must have ended
            time_exceeded = time_difference_seconds > 3600 * hour_threshold

            # is a route if back to start
            if distance <= definition_of_home and total_travelled > min_travel_distance:
                if len(current_route) >= 2:
                    trip_end = True
                # trip ends
                elif k < len(trip_frame) - 1:
                    # test if the next point is also at the start, then we end the trip
                    next_start_point = trip_frame.iloc[k + 1][
                        ["start_latitude", "start_longitude"]
                    ].values
                    distance_between_points = calc_distance(
                        start_point, next_start_point
                    )
                    if distance_between_points < definition_of_home:
                        trip_end = True

            if time_exceeded:
                # recursive check to see if we can route it more segmented
                ar = []
                if current_route and recurs == 0:
                    n_allow = get_car_starts(pd.DataFrame(current_route), start_frame)
                    i = 1
                    test += 1
                    if test > 10:
                        pass
                    elif (
                        current_route[-1].end_time - current_route[0].start_time
                    ).total_seconds() / 3600 > 72:
                        pass
                    else:
                        while True:
                            ar, _ = trip_extractor(
                                pd.DataFrame(current_route[i:]),
                                n_allow,
                                start_frame,
                                recurs=recurs + 1,
                                definition_of_home=definition_of_home,
                                hour_threshold=hour_threshold,
                                enforce_location=enforce_location,
                            )
                            i += 1
                            if ar or len(current_route) == i:
                                break
                if ar:
                    for a in ar:
                        if (
                            a["end_time"] - a["start_time"]
                        ).total_seconds() / 3600 > hour_threshold:
                            continue
                        a["from"] = "time"
                        all_routes.append(a)
                    current_route = []
                    trip_end = False
                    continue

            if trip_end:
                if time_exceeded:
                    continue

                all_routes.append(
                    route_format(
                        current_route,
                        car_id,
                        locat if pd.isna(enforce_location) else enforce_location,
                        total_travelled,
                    )
                )

                # reset parameters when trip has been saved or dismissed
                current_route = []
                trip_end = False

    return all_routes, time_sorted_frame


def trip_aggregator(
    car,
    c_trips,
    allowed_starts=None,
    hour_threshold=16.9,
    definition_of_home=0.2,
    min_travel_distance=0.5,
):
    """
    Function for handling the aggregation library.
    Called by the SkyHost - and FleetCompleteExtractor to easily aggregate the new trips. Handles the whole
    process.

    Parameters
    ----------
    car :   car object, with a defined id and location id
    c_trips :   dataframe, the car trips that will be sought to be aggregated into roundtrips
    allowed_starts  :   dataframe, the starts that are available to the car
    hour_threshold  :   int, the hour threshold for the aggregated trips
    definition_of_home  int, the distance allowed to a start
    min_travel_distance int, the minimum travel distance for a trip.

    Returns
    -------
    list of aggregated trips in dictionary.
    """
    if len(c_trips) == 0:
        return []
    print("Processing {} trips".format(c_trips.shape[0]))
    allowed_car_starts = get_car_starts(c_trips, allowed_starts)
    car_routes, trips_time = trip_extractor(
        c_trips,
        allowed_car_starts,
        allowed_starts,
        definition_of_home=definition_of_home,
        enforce_location=car.location,
        hour_threshold=hour_threshold,
        min_travel_distance=min_travel_distance
    )
    trip_routes = sum([len(a["ids"]) for a in car_routes])
    if trip_routes / len(c_trips) < 0.95:
        unused_trips, groups = get_consective_groups(trips_time, car_routes)
        if len(unused_trips) == 0:
            pass
        elif (
            unused_trips.iloc[-1].end_time - unused_trips.iloc[0].start_time
        ).days < 2:
            pass
        else:
            allow_skip_last = True
            if car_routes:
                allow_skip_last = (
                    True
                    if unused_trips.iloc[-1].end_time != car_routes[-1]["end_time"]
                    else False
                )

            unused_routed = [
                route_format(a, car.id, car.location)
                for k, b in enumerate(groups)
                for a in group_exceeded(
                    unused_trips.loc[b],
                    k + 1 == len(groups) and allow_skip_last,
                )
            ]
            car_routes += unused_routed
    if len(car_routes) == 0:
        return []
    car_routes_frame = pd.DataFrame(car_routes).sort_values(["start_time"])
    assert all(
        (
            car_routes_frame.start_time[1:].values
            - car_routes_frame.end_time[:-1].values
        )
        / np.timedelta64(1, "s")
        / 3600
        > 0
    ), "Overlapping routes"

    return car_routes_frame.to_dict("records")


def route_check(route):
    """
    Convenience function to check if a list of trips qualifies to a roundtrip.
    Checks that the distance of the trip is above .5 km and the duration is less than 16.9 hours.
    Criteria are hardcoded since the function is only called with roundtrips that are aggregated based on
    trips that could not be aggregated with trip_extractor.

    Parameters
    ----------
    route   :   list of frame objects that make up the

    Returns
    -------
    bool    :   roundtrip qualified

    """
    distance = sum([a.distance for a in route])
    route_duration = (route[-1].end_time - route[0].start_time).total_seconds() / 3600
    km_over_duration = distance / route_duration
    if distance > 0.5 and route_duration < 16.9:
        return True
    return False


def group_exceeded(group, skip_last=False):
    """
    Function for collecting un-aggregated trips. The trip_extractor returns the aggregated trips, from the original
    frame we can extract the trips logs that are not used in a qualified roundtrip. The logs that have adjacent
    logs will be sought to be grouped and added together in order to attribute these logs in the roundtrip frame.
    Otherwise, we end up having less trips in the dataset than what has actually be driven.
    Calls the route_check to check the coherence of the collection of trips.

    Iterates over the group, and adds the trip to a route if the log is started less than .75 hour from the last log

    Parameters
    ----------
    group   :   dataframe, the collected trips that are adjacent
    skip_last   :   bool, if the group is part of the last pulled trips on the major trip frame,
                    we skip adding as a route to allow the next aggregation job to properly aggregate these trips

    Returns
    -------
    list of routes
    """
    routes = []
    if len(group) > 0:
        last_time = group.iloc[0].end_time
        route = [group.iloc[0]]
    for k, a in enumerate(group.itertuples()):
        if k == 0:
            continue
        if (a.start_time - last_time).total_seconds() / 3600 > 0.75:
            if route_check(route):
                routes.append(route)
            route = []
        route.append(a)
        last_time = a.end_time

    if skip_last is False and route_check(route):
        routes.append(route)
    return routes


def inb(row, ts):
    """
    Checks that there are no overlapping timestamps. Sometimes the APIs return invalid data.

    Parameters
    ----------
    row :   dataframe row
    ts  :   times of the trips in route

    Returns
    -------
    False if everything is ok, else ids for overlapping
    """
    ovs = [
        k
        for k, a in enumerate(ts)
        if row["start_time"] >= a[0] and row["end_time"] <= a[1]
    ]
    if len(ovs):
        return ovs
    return False


def get_consective_groups(car_trips, routes):
    """
    Extracts the untripped trips from the car trips dataframe. The trip_extractor returns the aggregated trips,
    from the original frame we can extract the trips logs that are not used in a qualified roundtrip.
    The logs that have adjacent logs will be sought to be grouped and added together in order to attribute these logs
    in the roundtrip frame. Otherwise, we end up having less trips in the dataset than what has actually be driven.
    Parameters
    ----------
    car_trips   :   dataframe, the trips
    routes  :   routes, the aggregated routes

    Returns
    -------
    not_in  :   dataframe, the trips that are not used in a route
    groups  :   list, list of index groups that are adjacent and unused

    """
    times = [(a["start_time"], a["end_time"]) for a in routes]
    ids = [a for b in routes for a in b["ids"]]
    not_in = car_trips[~car_trips.id.isin(ids)].copy()
    not_in["overlap"] = not_in.apply(lambda x: inb(x, times), axis=1)
    not_in = not_in[not_in.overlap == False].copy()
    not_in["t_id"] = not_in.index.values
    diff = not_in.t_id.diff()
    groups = []
    group = []
    for tr, b in zip(not_in.itertuples(), diff):
        if pd.isna(b) is False and b != 1:
            groups.append(group)
            group = []
        group.append(tr.Index)

    if group:
        groups.append(group)

    return not_in, [a for a in groups if len(a) >= 2]


def route_format(route, car_id, location, distance=None):
    """
    Convenience function to convert to unified route format before saving to database

    Parameters
    ----------
    route   :   list of trips that makes up the route
    car_id  :   int, id of the car
    location    :   int, id of the location that should be enforced
    distance    :   int, the km distance of trip

    Returns
    -------
    dictionary of the trip
    """
    return {
        "start_time": route[0].start_time,
        "end_time": route[-1].end_time,
        "start_latitude": route[0].start_latitude,
        "start_longitude": route[0].start_longitude,
        "end_latitude": route[-1].end_latitude,
        "end_longitude": route[-1].end_longitude,
        "car_id": car_id,
        "distance": sum([a.distance for a in route]) if pd.isna(distance) else distance,
        "gps_points": [
            [
                (a.start_latitude, a.start_longitude),
                (a.end_latitude, a.end_longitude),
            ]
            for a in route
        ],
        "driver": None,
        "ids": [a.id for a in route],
        "start_location_id": location,
    }
