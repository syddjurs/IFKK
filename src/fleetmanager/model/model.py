import datetime
import operator
from itertools import groupby

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource
from sqlalchemy.orm.query import Query

from fleetmanager.data_access import AllowedStarts, RoundTrips, engine_creator
from fleetmanager.model import vehicle
from fleetmanager.model.qampo import qampo_simulation
from fleetmanager.model.qampo.classes import AlgorithmType
from fleetmanager.model.qampo.classes import Fleet as qampo_fleet
from fleetmanager.model.qampo.classes import Trip as qampo_trip
from fleetmanager.model.tco_calculator import TCOCalculator
from fleetmanager.model.vehicle import Bike, ElectricBike


class Trips:
    """Trips class for containing and manipulating trips of the simulation.

    Parameters
    ----------
    dataset : name (string) of dummy dataset to load. If dataset is a pandas DataFrame it is loaded as trips.all_trips.

    Attributes
    ----------
    all_trips : All trips in dataset (no filtering)
    trips : Trips in dataset applying date and department filter
    date_filter : boolean numpy array with same length as all_trips
    department_filter : boolean numpy array with same length as all_trips
    """

    def __init__(self, location=None, dataset=None, dates=None):
        self.engine = engine_creator()
        self.all_trips = []
        if isinstance(dataset, pd.DataFrame):
            self.all_trips = dataset
        else:
            query = Query([RoundTrips, AllowedStarts.address])
            if location:
                query = query.filter(RoundTrips.start_location_id == location)
            if dates:
                query = query.filter(
                    (RoundTrips.start_time > dates[0])
                    & (RoundTrips.end_time < dates[1])
                )
            self.all_trips = (
                pd.read_sql(
                    query.join(
                        AllowedStarts, RoundTrips.start_location_id == AllowedStarts.id
                    ).statement,
                    self.engine,
                )
                .sort_values(["start_time"])
                .reset_index()
                .iloc[:, 1:]
            )
            if len(self.all_trips) == 0:
                raise RuntimeError("Could not find any roundtrips")
            self.set_assignment()

        self.trips = self.all_trips.copy()
        self.distance_range = (
            self.trips.distance.min(),
            self.trips.distance.max(),
        )

    def set_assignment(self):
        """
        Method for setting up necessary columns on the trips in order
        to book-keep the allocated vehicles on the trips.
        """
        n = len(self.all_trips)
        self.all_trips["department"] = self.all_trips["address"].apply(
            lambda x: x.replace(",", "")
        )
        self.all_trips["tripid"] = self.all_trips["id"]
        self.all_trips["current"] = n * [vehicle.Unassigned()]
        self.all_trips["current_type"] = -np.ones((n,), dtype=int)
        self.all_trips["simulation"] = n * [vehicle.Unassigned()]
        self.all_trips["simulation_type"] = -np.ones((n,), dtype=int)

    def get_dummy_trips(self, n=10):
        """Helper function to generate fake data for testing"""
        tripid = 1 + np.arange(n)
        np.random.seed(42)
        distance = np.random.lognormal(1.9, 0.8, n)
        start = pd.Timestamp(
            year=2021, month=1, day=1, hour=3, minute=0
        ) + pd.to_timedelta(np.random.rand(n) * 25 * 24 * 60, unit="T")
        end = start + pd.to_timedelta(np.random.rand(n), unit="H")
        current = n * [vehicle.Unassigned()]
        current_type = -np.ones((n,), dtype=int)
        simulation = n * [vehicle.Unassigned()]
        simulation_type = -np.ones((n,), dtype=int)

        data = pd.DataFrame(
            {
                "car_id": np.random.randint(0, 5, n),
                "distance": distance,
                "start_time": start,
                "end_time": end,
                "start_latitude": np.random.random_sample(n) * 100,
                "start_longitude": np.random.random_sample(n) * 100,
                "end_latitude": np.random.random_sample(n) * 100,
                "end_longitude": np.random.random_sample(n) * 100,
                "start_location_id": np.random.randint(0, 44, n),
                "current": current,
                "current_type": current_type,
                "simulation": simulation,
                "simulation_type": simulation_type,
                "department": np.random.choice(["dept1", "dept2", "dept3"], n),
            }
        )

        data = data.sort_values(by="start_time")
        data["tripid"] = tripid
        return data

    def __iter__(self):
        """Yield trips as a single pandas row."""
        for i, r in self.trips.iterrows():
            yield r

    def _timestamp_to_timeslot(self, timestamp):
        """
        TODO: checkout this for a speed up: https://stackoverflow.com/questions/56796775/is-there-an-equivalent-to-numpy-digitize-that-works-on-an-pandas-intervalindex

        Parameters
        ----------
        timestamp : timestamp to be mapped to a timeslot

        Returns
        -------
        i : timeslot index as int. If None is returned the timestamp is outside the timeslots.
        """

        time_indexes = np.nonzero(
            np.logical_and(
                self.timestamps.start_time <= timestamp,
                self.timestamps.end_time > timestamp,
            )
        )
        return time_indexes[0] if len(time_indexes) > 0 else None

    def set_timestamps(self, timestamps):
        """
        Set timestamps of Trips and mapped all trips to the corresponding timeslots.

        Parameters
        ----------
        timestamps : pandas PeriodIndex
        """
        self.timestamps = timestamps
        start_slot = []
        end_slot = []
        x = 0
        for i, roundtrip in enumerate(self.__iter__()):
            x += 1
            start_slot.append(self._timestamp_to_timeslot(roundtrip.start_time))
            end_slot.append(self._timestamp_to_timeslot(roundtrip.end_time))

        self.trips["start_slot"] = start_slot
        self.trips["end_slot"] = end_slot

    def set_filtered_trips(self):
        """
        Applies date and department filter and sets model.trips accordingly.
        """
        # this function should create the filtered trips from filters
        self.trips = self.all_trips.copy()
        self.distance_range = (
            self.trips.distance.min(),
            self.trips.distance.max(),
        )


class ConsequenceCalculator:
    """
    ConsequenceCalculator class for computing economical, transport and emission consequences of a simulation


    Attributes

    ----------

    capacity_source : bokeh ColumnDataSource for containing capacity computed by the simulation

    consequence_table : bokeh ColumnDataSource for containing consequences computed by the simulation

    """

    def __init__(self, timeframe=None, states=None):
        """
        Initiates the class with default timeframe on 30 days with default states on "current" and "simulation".
        The states will be used by the compute method to iterate over the elements in order to calculate the
        consequences.
        Parameters
        ----------
        timeframe   :   list of datetimes - [start time, end time] - defaults to [a month ago, now]
        states      :   list of strings - default to ["current", "simulation"]
        """
        if states is None:
            states = ["current", "simulation"]
        if timeframe is None:
            now = datetime.datetime.now()
            amonthback = now - datetime.timedelta(days=30)
            self.timeframe = [amonthback, now]
        self.table_keys = [
            "CO2-udledning [kg]",
            "Antal ture uden køretøj",
            "Udbetalte kørepenge [kr]",
            "Årlig gns. omkostning [kr/år]",
            "POGI årlig brændstofforbrug [kr/år]",
            "POGI CO2-ækvivalent udledning [CO2e]",
            "POGI samfundsøkonomiske omkostninger [kr/år]",
            "Samlet omkostning [kr/år]",
        ]

        self.states = states
        self.values = {
            f"{state[:3]}": [0] * len(self.table_keys) for state in self.states
        }

        self.consequence_table = ColumnDataSource()

        # capacity
        for state in states:
            setattr(
                self,
                f"{state}_capacity",
                {"unassigned_trips": [0, 0], "trip_capacity": [0, 0]},
            )
            setattr(self, f"{state[:3]}_allowance", 0)

        self.capacity_source = ColumnDataSource()

        self.update_consequence_table()
        self.update_capacity_source()

    def update_consequence_table(self):
        """Update the consequence table"""
        self.consequence_table.data = {
            "keys": self.table_keys,
        }
        for state in self.states:
            self.consequence_table.data[f"{state[:3]}_values"] = getattr(
                self, "values"
            )[f"{state[:3]}"]

    def update_capacity_source(self):
        d = {"timeframe": self.timeframe}

        for state in self.states:
            c_name = state[:3]
            d[f"{c_name}_unassigned_trips"] = getattr(self, f"{state}_capacity")[
                "unassigned_trips"
            ]
            d[f"{c_name}_trip_capacity"] = getattr(self, f"{state}_capacity")[
                "trip_capacity"
            ]
        self.capacity_source.data = d

    def compute(self, simulation, drivingallowance, tco_period):
        """
        The compute function that calculate the consequences;
        1) explicit CO2 emission,
            Calculated by taking the product of each vehicle's allocated km to a yearly approximation
            and the vehicle's explict noted gram CO2-emission pr. kilometer. Since this is only relevant
            for fossile vehicles, we don't report this number because it would always show the electrical vehicles
            to have 0 emission. Hence, we refer to 6) yearly CO2-e.
        2) number of trips without vehicle,
            Sum of all trips that have no vehicle assigned. Displayed in the simulation.trips.[inventory_type_column]
            with a value of -1.
        3) pay out in driving allowance,
            In order to punish the unallocated trips driving allowance is simulated. All unallocated trips are summed
            to a yearly approximation. The driving allowance is paid in rates; 3.44 kr. pr. km. under 840 kilometer
            threshold and 1.90 kr. pr. km. above 840 kilometer threshold.
        4) yearly average expense on hardware
            Is calculated by taking the sum of the reported "omkostning_aar" for all vehicles
        5) yearly expense on fuel
            Is calculated through the tco_calcluate.TCOCalculator, which is based on the tool
            "tco-vaerktoej-motorkoeretoejer" from POGI. Check the details on the class.
        6) yearly CO2-e expense (implicit CO2 emission)
            Is calculated through the tco_calcluate.TCOCalculator, which is based on the tool
            "tco-vaerktoej-motorkoeretoejer" from POGI. Check the details on the class.
        7) total yearly expense
            Is calculated by taking the sum of driving allowance, yearly average expense on hardware and yearly expense
            on fuel.

        Parameters
        ----------
        simulation  :   model.Simulation class - the simulation class with it's associated trips. The inventory - and distance columns of the
                            simulation.trips frame holds the necessary data to calculate the aforementioned values.
        drivingallowance    :   model.DrivingAllowance - a DrivingAllowance class or None.
        tco_period  :   list of two ints ([0, 1]) - the selected tco_period which is passed to the TCO_Calculator object.
                            First int to define projection periode, second int to define the evaluation period.

        Returns
        -------

        """
        if drivingallowance is None:
            drivingallowance = DrivingAllowance()
        self.timeframe = [
            simulation.trips.trips.start_time.min(),
            simulation.trips.trips.end_time.max(),
        ]
        days = (self.timeframe[1] - self.timeframe[0]).total_seconds() / 3600 / 24
        if days <= 0:
            # we have to assume that there's at least one day worth of data
            days = 1

        calculate_this = {
            key: {val: 0 for val in self.table_keys} for key in self.states
        }

        vehicles_used = {key: {} for key in self.states}
        for roundtrip in simulation.trips:
            # co2 udledning
            # record the vehicle and how much it spent
            for state in self.states:
                if getattr(roundtrip, f"{state}_type") == -1:
                    pass
                elif getattr(roundtrip, state) not in vehicles_used[state]:
                    co2_pr_km = (
                        0
                        if pd.isna(roundtrip[state].co2_pr_km)
                        else roundtrip[state].co2_pr_km
                    )
                    vehicles_used[state][getattr(roundtrip, state)] = roundtrip.distance
                    calculate_this[state]["CO2-udledning [kg]"] += (
                        co2_pr_km * roundtrip.distance
                    )
                else:
                    co2_pr_km = (
                        0
                        if pd.isna(roundtrip[state].co2_pr_km)
                        else roundtrip[state].co2_pr_km
                    )
                    vehicles_used[state][
                        getattr(roundtrip, state)
                    ] += roundtrip.distance
                    calculate_this[state]["CO2-udledning [kg]"] += (
                        co2_pr_km * roundtrip.distance
                    )

        for state in self.states:
            c_name = state[:3]

            # antal ture uden køretøj
            calculate_this[state]["Antal ture uden køretøj"] = (
                simulation.trips.trips[f"{state}_type"] == -1
            ).sum()

            # straf ukørte ture
            undriven = simulation.trips.trips[
                simulation.trips.trips[f"{state}_type"] == -1
            ]
            undriven_km = undriven.distance.sum()
            undriven_yearly = undriven_km / days * 365

            # udbetalte kørepenge
            allowance = drivingallowance.calculate_allowance(undriven_yearly)

            # udledning
            undriven_tco = TCOCalculator(
                koerselsforbrug=undriven_yearly,
                drivmiddel="benzin",
                bil_type="benzin",
                antal=1,
                evalueringsperiode=1,
                fremskrivnings_aar=tco_period[0],
                braendstofforbrug=20,
            )
            co2e_undriven, samfund_undriven = undriven_tco.ekstern_miljoevirkning(
                sum_it=True
            )
            calculate_this[state][
                "POGI CO2-ækvivalent udledning [CO2e]"
            ] += co2e_undriven
            calculate_this[state][
                "POGI samfundsøkonomiske omkostninger [kr/år]"
            ] += samfund_undriven

            # årlig gns. omkostning
            yearly_cost = sum(
                v.omkostning_aar
                for v in getattr(simulation.fleet_manager, f"{state}_fleet")
            )
            calculate_this[state]["Årlig gns. omkostning [kr/år]"] = yearly_cost

            # pogi årlig brændstofforbrug
            # pogi co2-ækvivalent udledning
            # pogi samfundsøkonomiske omkostninger
            for vehicle, distance in vehicles_used[state].items():
                distance_yearly = distance / days * 365
                vehicle_tco = TCOCalculator(
                    koerselsforbrug=distance_yearly,
                    drivmiddel=vehicle.fuel,
                    bil_type=vehicle.fuel,
                    antal=1,
                    evalueringsperiode=1,  # tco_period[1],
                    fremskrivnings_aar=tco_period[0],
                    braendstofforbrug=vehicle.wltp_fossil,
                    elforbrug=vehicle.wltp_el,
                )
                co2e, samfund = vehicle_tco.ekstern_miljoevirkning(sum_it=True)
                driftsomkostning = vehicle_tco.driftsomkostning
                calculate_this[state][
                    "POGI årlig brændstofforbrug [kr/år]"
                ] += driftsomkostning
                calculate_this[state][
                    "POGI samfundsøkonomiske omkostninger [kr/år]"
                ] += samfund
                calculate_this[state]["POGI CO2-ækvivalent udledning [CO2e]"] += co2e

            # compute capacity
            # for each day, compute number of trips
            sub = simulation.trips.trips[
                ["start_time"] + [f"{state}_type" for state in self.states]
            ].copy(deep=True)

            sub[f"{c_name}_unassigned"] = sub[f"{state}_type"] == -1
            resampled = sub.resample("D", on="start_time")[
                [f"{c_name}_unassigned"]
            ].sum()
            self.timeframe = resampled.index.to_pydatetime()
            n = len(self.timeframe)
            getattr(self, f"{state}_capacity")["unassigned_trips"] = list(
                getattr(resampled, f"{c_name}_unassigned")
            )
            getattr(self, f"{state}_capacity")["trip_capacity"] = n * [0]

            calculate_this[state]["Samlet omkostning [kr/år]"] = (
                allowance
                + calculate_this[state]["Årlig gns. omkostning [kr/år]"]
                + calculate_this[state]["POGI årlig brændstofforbrug [kr/år]"]
            )

            getattr(self, "values")[c_name] = [
                calculate_this[state][key] for key in self.table_keys
            ]
            getattr(self, "values")[c_name][2] = allowance

        # update sources for frontend
        self.update_consequence_table()
        self.update_capacity_source()

    def get_html_information(self):
        """Function for producing html used on infromation panel"""

        html = "<h2>Konsekvensberegning</h2>"
        html += "<table ><tr><th>Konsekvensmål</th><th>Nuværende værdi</th><th>Simuleret værdi</th></tr>"
        for t in zip(self.table_keys, self.cur_vals, self.sim_vals):
            html += f"<tr><td>{t[0]}</td><td>{t[1]:.02f}</td><td>{t[2]:.02f}</td></tr>"
        html += "</table>"
        return html


class FleetManager:
    """FleetManager class keeps track of the fleets and the booking.

    parameters
    ----------
    options: options of type model.OptionsFile

    attributes
    ----------
    vehicle_factory : types of vehicles in fleet of type vehicle.VehicleFactory
    simulation_fleet : simulation fleet of type vehicle.FleetInventory
    current_fleet : current fleet of type vehicle.FleetInventory

    """

    def __init__(self):
        # set the available vehicles
        self.vehicle_factory = vehicle.VehicleFactory()

        # initialise empty fleets
        self.simulation_fleet = vehicle.FleetInventory(
            self.vehicle_factory, name="simulation"
        )
        self.current_fleet = vehicle.FleetInventory(
            self.vehicle_factory, name="current"
        )

    def set_timestamps(self, ts):
        """Set timestamps of fleets"""
        self.simulation_fleet.set_timestamps(ts)
        self.current_fleet.set_timestamps(ts)

    def set_current_fleet(self, bikes=0, ebikes=0, ecars=0, cars=0):
        """Set current fleet composition"""
        self.current_fleet.bikes = bikes
        self.current_fleet.ebikes = ebikes
        self.current_fleet.ecars = ecars
        self.current_fleet.cars = cars
        self.current_fleet.initialise_fleet()

    def set_simulation_fleet(self, bikes=0, ebikes=0, ecars=0, cars=0):
        """Set simulation fleet composition"""
        self.simulation_fleet.bikes = bikes
        self.simulation_fleet.ebikes = ebikes
        self.simulation_fleet.ecars = ecars
        self.simulation_fleet.cars = cars
        self.simulation_fleet.initialise_fleet()

    def get_html_information(self):
        """Function for producing html used on infromation panel"""

        cars = {
            vehicle_: getattr(self.current_fleet, vehicle_)
            for vehicle_ in self.current_fleet.vehicle_types
            if self.vehicle_factory.vmapper[vehicle_].vehicle_type_number == 0
        }
        ecars = {
            vehicle_: getattr(self.current_fleet, vehicle_)
            for vehicle_ in self.current_fleet.vehicle_types
            if self.vehicle_factory.vmapper[vehicle_].vehicle_type_number == 1
        }
        ebikes = {
            vehicle_: getattr(self.current_fleet, vehicle_)
            for vehicle_ in self.current_fleet.vehicle_types
            if self.vehicle_factory.vmapper[vehicle_].vehicle_type_number == 2
        }
        bikes = {
            vehicle_: getattr(self.current_fleet, vehicle_)
            for vehicle_ in self.current_fleet.vehicle_types
            if self.vehicle_factory.vmapper[vehicle_].vehicle_type_number == 3
        }
        sim_cars = {
            vehicle_: getattr(self.simulation_fleet, vehicle_)
            for vehicle_ in self.simulation_fleet.vehicle_types
            if self.vehicle_factory.vmapper[vehicle_].vehicle_type_number == 0
        }
        sim_ecars = {
            vehicle_: getattr(self.simulation_fleet, vehicle_)
            for vehicle_ in self.simulation_fleet.vehicle_types
            if self.vehicle_factory.vmapper[vehicle_].vehicle_type_number == 1
        }
        sim_ebikes = {
            vehicle_: getattr(self.simulation_fleet, vehicle_)
            for vehicle_ in self.simulation_fleet.vehicle_types
            if self.vehicle_factory.vmapper[vehicle_].vehicle_type_number == 2
        }
        sim_bikes = {
            vehicle_: getattr(self.simulation_fleet, vehicle_)
            for vehicle_ in self.simulation_fleet.vehicle_types
            if self.vehicle_factory.vmapper[vehicle_].vehicle_type_number == 3
        }

        tmapper = {
            "Car": "Fossilbil",
            "ElectricCar": "Elbil",
            "Bike": "Cykel",
            "ElectricBike": "Elcykel",
        }

        html = "<h2>Flådeinformation</h2>"
        html += f"<dl><dt>Flådesammensætning, nuværende</dt><dd>Fossilbiler: {sum(cars.values())}</dd><dd>Elbiler: {sum(ecars.values())}</dd><dd>Cykler: {sum(bikes.values())}</dd><dd>Elcykler: {sum(ebikes.values())}</dd></dl>"
        for (vtype, vehicles) in (
            ("Car", cars),
            ("ElectricCar", ecars),
            ("Bike", bikes),
            ("ElectricBike", ebikes),
        ):
            html += f"<dl><dt>Køretøjstype: {tmapper[vtype]}:</dt>"

            for key, val in vehicles.items():
                html += f"<dd>{key}: {val}</dd>"
            html += "<dl/>"

        html += f"<br><br><dl><dt>Flådesammensætning, simuleret</dt><dd>Fossilbiler: {sum(sim_cars.values())}</dd><dd>Elbiler: {sum(sim_ecars.values())}</dd><dd>Cykler: {sum(sim_bikes.values())}</dd><dd>Elcykler: {sum(sim_ebikes.values())}</dd></dl><br>"
        for (vtype, vehicles) in (
            ("Car", sim_cars),
            ("ElectricCar", sim_ecars),
            ("Bike", sim_bikes),
            ("ElectricBike", sim_ebikes),
        ):
            html += f"<dl><dt>Køretøjstype: {tmapper[vtype]}:</dt>"

            for key, val in vehicles.items():
                html += f"<dd>{key}: {val}</dd>"
            html += "<dl/>"
        return html


class Simulation:
    """
    The major Simulation class for performing simulation on trips.

    parameters
    ----------
    trips : trips for simulation of type modelTrips
    fleet_manager : fleet manager for handling fleets of type model.FleetManager
    progress_callback : None
    tabu    :   bool - to let the simulation know if it's a tabu simulation. If so, only the simulation setup will be
                simulated, and not the current.
    intelligent_simulation  :   bool - should intelligent simulation be used, i.e. Qampo algorithm to allocate trips.
    timestamp_set   :   bool -  whether the simulation trips already have generated timeslots

    """

    def __init__(
        self,
        trips,
        fleet_manager,
        progress_callback,
        tabu=False,
        intelligent_simulation=False,
        timestamps_set=False,
    ):
        self.trips = trips
        self.fleet_manager = fleet_manager
        self.progress_callback = progress_callback
        self.tabu = tabu

        self.useQampo = intelligent_simulation

        if timestamps_set is False:
            self.time_resolution = pd.Timedelta(minutes=1)
            start_day = self.trips.trips.start_time.min().date()
            end_day = self.trips.trips.end_time.max().date() + pd.Timedelta(days=1)
            self.timestamps = pd.period_range(start_day, end_day, freq=self.time_resolution)
            self.trips.set_timestamps(self.timestamps)

        # dummy vehicle for unassigned trips
        self.unassigned_vehicle = vehicle.Unassigned(name="Unassigned")

    def run(self):
        """Runs simulation of current and simulation fleet"""
        # push timetable to vehicle fleet
        self.fleet_manager.set_timestamps(self.timestamps)

        if self.useQampo:
            if self.tabu:
                self.run_single_qampo(self.fleet_manager)
            else:
                self.run_single_qampo(self.fleet_manager.simulation_fleet)
                self.run_single(self.fleet_manager.current_fleet)
        else:
            if self.tabu:
                self.run_single(self.fleet_manager)
            else:
                self.run_single(self.fleet_manager.simulation_fleet)
                self.run_single(self.fleet_manager.current_fleet)

    def run_single_qampo(self, fleet_inventory, algorithm_type="exact_mip"):
        """Convenience function for running simualtion on a single fleet through qampo api.

        parameters
        ----------
        fleet_inventory : fleet inventory to run simualtion on. Type model.FleetInventory.
        algorithm_type : the algorithm the qampo api uses. must be either 'exact_mip', 'greedy' or 'exact_cp'
        """
        # setting up api parameters
        # Helper function: Changes start day to be 00:00 of end day if more time is spend driving in end day
        if self.trips.trips.iloc[-1].name != len(self.trips.trips) - 1:
            raise IndexError(
                "Some initial trips were falsely filtered after re-indexing."
            )

        bike_fleet = fleet_inventory.copy_bike_fleet("bike_fleet")
        bike_fleet.set_timestamps(self.timestamps)
        self.run_single(bike_fleet)

        def set_start_times(trip):
            if trip["start_time"].normalize() != trip["end_time"].normalize():
                time_in_start_day = trip["end_time"].normalize() - trip["start_time"]
                time_in_end_day = trip["end_time"] - trip["end_time"].normalize()
                if time_in_start_day < time_in_end_day:
                    trip["start_time"] = trip["end_time"].normalize()
            return trip

        trips_day_fixed = map(
            lambda trip: set_start_times(trip),
            self.trips.trips[self.trips.trips.bike_fleet_type == -1].to_dict("records"),
        )
        trips_day_fixed = sorted(trips_day_fixed, key=operator.itemgetter("start_time"))

        # Splitting trips into distinct days as the api can only work on a single day at a time

        trips_pr_day = []
        for k, g in groupby(
            trips_day_fixed, lambda trip: trip["start_time"].normalize()
        ):
            trips_pr_day.append(list(g))

        response = []
        for trips_single_day in trips_pr_day:
            data = self.generate_qampo_data(fleet_inventory, trips_single_day)
            fleet = qampo_fleet(**data["fleet"])
            trips = list(map(lambda T: qampo_trip(**T), data["trips"]))
            simulation = qampo_simulation.optimize_single_day(
                fleet, trips, AlgorithmType.EXACT_MIP
            )
            response.append(simulation)

        # Booking vehicles in accordance to the result from qampo api
        trip_vehicle = [[]] * len(self.trips.trips)
        trip_vehicle_type = [[]] * len(self.trips.trips)
        for content in response:
            for assignment in content.assignments:
                id = assignment.vehicle.id
                v = next(filter(lambda v: v.vehicle_id == id, fleet_inventory))
                for t in assignment.route.trips:
                    trip = next(filter(lambda tt: tt["tripid"] == t.id, self.trips))
                    v.book_trip(trip)
                    trip_vehicle[trip.name] = v
                    trip_vehicle_type[trip.name] = v.vehicle_type_number

        for k in range(len(trip_vehicle)):
            if type(trip_vehicle[k]) is list:
                trip_vehicle[k] = self.trips.trips.bike_fleet[k]
                trip_vehicle_type[k] = self.trips.trips.bike_fleet_type[k]

        self.trips.trips[fleet_inventory.name] = trip_vehicle
        self.trips.trips[fleet_inventory.name + "_type"] = trip_vehicle_type

    def generate_qampo_data(self, fleet_inventory, trips):
        """Convenience function for converting fleet inventory and trips data to json format
        required by qampo api.

        parameters
        ----------
        fleet_inventory : fleet inventory to run simualtion on. Type model.FleetInventory.
        trips: trip data to run simulation on.
        """
        data = {
            "fleet": {
                "vehicles": [],
                # Needs to not be hard coded
                "employee_car": {
                    "variable_cost_per_kilometer": 20.0,
                    "co2_emission_gram_per_kilometer": 400.0,
                },
                "emission_cost_per_ton_co2": 5000.0,
            },
            "trips": [],
        }

        for v in fleet_inventory:
            if v.vehicle_type_number in [2, 3]:
                # skip the bikes as we handle those
                continue
            vehicle = {
                "id": int(v.vehicle_id),
                "name": v.name,
                "range_in_kilometers": float(v.max_distance_per_day),
                "variable_cost_per_kilometer": v.vcprkm,
                "maximum_driving_in_minutes": 1440
                if pd.isna(v.sleep)
                else (24 - v.sleep) * 60,
                "co2_emission_gram_per_kilometer": v.qampo_gr,
            }
            data["fleet"]["vehicles"].append(vehicle)

        for t in trips:
            trip = {
                "id": int(t["tripid"]),
                "start_time": t["start_time"].strftime("%Y-%m-%dT%H:%M:%S"),
                "end_time": t["end_time"].strftime("%Y-%m-%dT%H:%M:%S"),
                "length_in_kilometers": float(round(t["distance"], 2)),
            }
            data["trips"].append(trip)

        return data

    def run_single(self, fleet_inventory):
        """Convenience function for running simualtion on a single fleet

        Takes the fleet and iterates over the trips to see which, if any, vehicle is available for booking.
        If the fleet_inventory name is current, the vehicles are booked according to its recorded trips.
        This will overwrite any rules implied by the simulation, e.g. vehicle cannot be booked for a trip on the
        same minute stamp as it ends a trips, sleep rules for electrical cars etc.

        The vehicles should be sorted according to the desired priority (defaults to co2 emission). For every trip the
        first available vehicle is booked for the trips.

        parameters
        ----------
        fleet_inventory : fleet inventory to run simualtion on. Type model.FleetInventory.
        """
        # loop over trips
        trip_vehicle = []
        trip_vehicle_type = []
        flagged = []
        for t in self.trips:
            booked_real = False
            if fleet_inventory.name == "current":
                # overwrites the simulated booking to reflect "reality"
                if any([str(a.id) == str(t.car_id) for a in fleet_inventory]):
                    for v in fleet_inventory:
                        if str(v.id) == str(t.car_id):
                            booked, acc, avail = v.bypass_book(t)  # v.book_trip(t)

                            if booked:
                                trip_vehicle.append(v)
                                trip_vehicle_type.append(v.vehicle_type_number)
                                booked_real = True
                                break
                else:
                    # the car that drove the trip in real life is not part of the selected "current" fleet.
                    if t.car_id not in flagged:
                        print(
                            f"********** car id from trips not in {str(t.car_id)}",
                            flush=True,
                        )
                        flagged.append(t.car_id)

            if booked_real:
                continue
            # loop over vehicles and check for availability
            booked = False
            for v in fleet_inventory:
                booked, acc, avail = v.book_trip(t)

                if booked:
                    trip_vehicle.append(v)
                    trip_vehicle_type.append(v.vehicle_type_number)
                    break

            if not booked:
                trip_vehicle.append(self.unassigned_vehicle)
                trip_vehicle_type.append(self.unassigned_vehicle.vehicle_type_number)

        # add vehicles to trips
        self.trips.trips[fleet_inventory.name] = trip_vehicle
        self.trips.trips[fleet_inventory.name + "_type"] = trip_vehicle_type

    def __str__(self):
        return str(self.trips)


class DrivingAllowance:
    """Class for containing and manipulating driving allowance."""

    def __init__(self):
        self.allowance = {"low": 3.44, "high": 1.90}
        # TODO: make a this editable in ui
        self.distance_threshold = 840

    def __str__(self):
        return (
            f"Driving allowance {self.allowance}\n  Dist: {self.distance_threshold}\n"
        )

    def calculate_allowance(self, yearly_distance):
        """
        Method for calculating the driving allowance for unallocated trips. Defines a threshold of 840 km which is
        eligible to get the high allowance fee, from which the fee drops to the low allowance. Especially useful in
        tabu search in order not to favor unallocated trips because it is cheap.

        Parameters
        ----------
        yearly_distance :   int - sum of kilometers without an allocated vehicle

        Returns
        -------
        driving allowance   :   int - sum of money paid out in driving allowance to attribute the unallocated trips
        """
        if yearly_distance > self.distance_threshold:
            allowance_to_pay = sum(
                [
                    self.distance_threshold * self.allowance["low"],
                    (yearly_distance - self.distance_threshold)
                    * self.allowance["high"],
                ]
            )
        else:
            allowance_to_pay = yearly_distance * self.allowance["low"]
        return allowance_to_pay


class Model:
    """Model class for MVC pattern of the simulation tool.

    Parameters
    ----------
    location    :   int - id of the location selected for the simulation
    dates   :   list of datetime - the selected time frame for the trips to simulated - i.e. [start time, end time]
                will define the period from which the trips will be pulled.

    """

    def __init__(self, location=None, dates=None, tco_period=(0, 1)):
        """
        Method for handling all interacting classes.
        Essential elements to be loaded are:
        trips   :   Trips class - holding all information on the trips from defined filters (location, dates)
        fleet_manager   :   FleetManager class - to hold the current - and simulation fleet will initialise vehicle
                                objects
        consequence_calculator  :   ConsequenceCalculator class - to associate the simulation with
                                    the simulation results
        drivingallowance    :   DrivingAllowance class - to attribute unallocated trips with associated inventory_type
                                value -1

        Parameters
        ----------
        location    :   int - id of the location selected for the simulation
        dates   :   list of datetime - the selected time frame for the trips to simulated - i.e. [start time, end time]
                    will define the period from which the trips will be pulled.
        tco_period  : tuple or list of two ints defining the projection period and evaluation period of the
                        TCO calculation
        """
        # todo make dates a controllable entry parameter
        self.trips = Trips(location=location, dates=dates)

        self.fleet_manager = FleetManager()

        self.consequence_calculator = ConsequenceCalculator()

        # static references to data sources needed by the view
        self.consequence_source = self.consequence_calculator.consequence_table
        self.capacity_source = self.consequence_calculator.capacity_source
        self.progress_source = ColumnDataSource(
            data={"start": [0.0], "progress": [0.0]}
        )
        self.progress_callback = lambda x: print(f"Simulér ({100 * x}%)")

        # update histogram sources
        self.current_hist_datasource = ColumnDataSource()
        self.simulation_hist_datasource = ColumnDataSource()
        self.compute_histogram()

        # driving allowance
        self.drivingallowance = DrivingAllowance()
        self.tco_period = tco_period

    def _update_progress(self, progress):
        """Tester function for updating progress of simualtion"""
        self.progress_source.data = {"start": [0.0], "progress": [progress]}
        if progress > (1.0 - 1e-12):
            self.progress_callback(False)
        else:
            self.progress_callback(True)

    def _update_progress_stdout(self, progress):
        print(progress)

    def run_simulation(
        self,
        intelligent_simulation,
        bike_max_distance=5,
        bike_time_slots=None,
        max_bike_time_slot=0,
        bike_percentage=100,
        km_aar=False,
    ):
        """
        Create and run a simulation. Updates histograms and consequence information.
        Sets up the simulation and initialises the fleets and runs the simulation.

        Parameters
        ------------
        intelligent_simulation  :   bool - to be passed to the simulation object
        bike_max_distance   :   int - to define bike configuration, max allowed distance for a bike trip
        bike_time_slots :   bike configuration time slot, when are bike vehicles allowed to accept trips
        max_bike_time_slot  :   bike configuration, how many bike slots are available for bikes
        bike_percentage :   how many percentage of the trips that qualifies for bike trip should be accepted
        km_aar  :   bool - should the vehicles associated km_aar constrain the vehicle from accepting trips when the
                        yearly capacity is reached. Only available on intelligent_simulation = False

        """
        if bike_time_slots is None:
            bike_time_slots = []
        self.simulation = Simulation(
            self.trips,
            self.fleet_manager,
            self._update_progress,
            intelligent_simulation=intelligent_simulation,
        )

        Bike.max_distance_pr_trip = bike_max_distance
        ElectricBike.max_distance_pr_trip = bike_max_distance
        Bike.allowed_driving_time_slots = bike_time_slots
        ElectricBike.allowed_driving_time_slots = bike_time_slots
        Bike.max_time_slot = max_bike_time_slot
        ElectricBike.max_time_slot = max_bike_time_slot
        Bike.percentage = bike_percentage
        ElectricBike.percentage = bike_percentage

        # collect data from frontend
        self.simulation.fleet_manager.current_fleet.initialise_fleet(km_aar)
        self.simulation.fleet_manager.simulation_fleet.initialise_fleet(km_aar)

        self.simulation.run()

        # update data sources for frontend
        self.compute_histogram()

        # update consequence sources for frontend
        self.consequence_calculator.compute(
            self.simulation, self.drivingallowance, self.tco_period
        )

    def compute_histogram(self, mindist=0, maxdist=None):
        """Compute histograms for current and simulation

        parameters
        ----------
        mindist : defaults to 0. Minimum distance to use for histograms
        maxdist : Maximum distance to use for histograms. If None, use the maximum distance of the trips

        """
        if maxdist is None:
            maxdist = self.trips.trips.distance.max()

        delta = (maxdist - mindist) / 20.0
        delta = max(delta, 0.001)
        distance_edges = np.arange(mindist, maxdist, delta)

        self.current_hist = {"edges": distance_edges[:-1]}
        self.simulation_hist = {"edges": distance_edges[:-1]}

        for i in range(-1, 4):
            # current
            d = self.trips.trips.distance[self.trips.trips.current_type == i]
            counts, edges = np.histogram(d, bins=distance_edges)
            self.current_hist[vehicle.vehicle_mapping[i]] = counts

            # simulation
            d = self.trips.trips.distance[self.trips.trips.simulation_type == i]
            counts, edges = np.histogram(d, bins=distance_edges)
            self.simulation_hist[vehicle.vehicle_mapping[i]] = counts

        self.current_hist_datasource.data = self.current_hist
        self.simulation_hist_datasource.data = self.simulation_hist
