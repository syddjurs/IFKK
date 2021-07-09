import base64
import pathlib
import datetime
import numpy as np
import pandas as pd
import xlwings as xw

from bokeh.models import ColumnDataSource

import vehicle
from utils import data_from_db


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

    def __init__(self, dataset="dummy"):

        if isinstance(dataset, pd.DataFrame):
            self.all_trips = dataset
        else:
            if dataset == "dummy":
                self.all_trips = self.get_dummy_trips()
            elif dataset == "dummy_outside":
                self.all_trips = self.get_dummy_outside()
            elif dataset == "empty":
                self.all_trips = self.get_dummy_empty()
            else:
                raise ValueError

        # init filters
        self.date_filter = np.full((self.all_trips.shape[0],), True)
        self.department_filter = np.full((self.all_trips.shape[0],), True)

        self.set_filtered_trips()

    def get_dummy_empty(self):
        """Helper function to generate fake data for testing"""
        data = pd.DataFrame(
            {
                "car_id": [],
                "distance": [],
                "starttime": [],
                "endtime": [],
                "start_latitude": [],
                "start_longitude": [],
                "current": [],
                "current_type": [],
                "simulation": [],
                "simulation_type": [],
                "department": [],
            }
        )
        return data

    def get_dummy_trips(self):
        """Helper function to generate fake data for testing"""
        n = 10
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
                "starttime": start,
                "endtime": end,
                "start_latitude": np.random.random_sample(n) * 100,
                "start_longitude": np.random.random_sample(n) * 100,
                "current": current,
                "current_type": current_type,
                "simulation": simulation,
                "simulation_type": simulation_type,
                "department": np.random.choice(["dept1", "dept2", "dept3"], n),
            }
        )

        data = data.sort_values(by="starttime")
        data["tripid"] = tripid
        return data

    def get_dummy_outside(self):
        """Helper function to generate fake data for testing"""
        n = 3
        tripid = 1 + np.arange(n)
        distance = np.random.lognormal(1.8, 1.2, n)
        # start = pd.Timestamp(year=2021, month=1, day=1, hour=3, minute=0) + pd.to_timedelta(np.random.rand(n) * 24 * 60, unit='T') # pd.to_timedelta(np.random.rand(n) * 10, unit='H')
        # end = start + pd.to_timedelta(np.random.rand(n), unit='H')
        start = [
            pd.Timestamp(year=2021, month=1, day=5, hour=6, minute=0),
            pd.Timestamp(year=2021, month=1, day=1, hour=9, minute=0),
            pd.Timestamp(year=2021, month=1, day=1, hour=12, minute=0),
        ]
        end = [
            pd.Timestamp(year=2021, month=1, day=5, hour=7, minute=0),
            pd.Timestamp(year=2021, month=1, day=1, hour=10, minute=0),
            pd.Timestamp(year=2021, month=1, day=2, hour=6, minute=0),
        ]
        current = n * [vehicle.Unassigned()]
        simulation = n * [vehicle.Unassigned()]
        current_type = -np.ones((n,), dtype=int)
        simulation_type = -np.ones((n,), dtype=int)

        department = np.random.choice(["dept1", "dept2", "dept3"], n)
        car_id = np.random.randint(0, 5, n)
        data = pd.DataFrame(
            {
                "car_id": car_id,
                "distance": distance,
                "starttime": start,
                "endtime": end,
                "current": current,
                "current_type": current_type,
                "simulation": simulation,
                "simulation_type": simulation_type,
                "department": department,
            }
        )

        data = data.sort_values(by="starttime")
        data["tripid"] = tripid
        return data

    def __iter__(self):
        """Yield trips as a single pandas row."""
        for i, r in self.trips.iterrows():
            yield r

    def _timestamp_to_timeslot(self, timestamp):
        """TODO: Docstring for data_from_db.

        TODO: checkout this for a speed up: https://stackoverflow.com/questions/56796775/is-there-an-equivalent-to-numpy-digitize-that-works-on-an-pandas-intervalindex

        Parameters
        ----------
        timestamp : timestamp to be mapped to a timeslot

        Returns
        -------
        i : timeslot index as int. If None is returned the timestamp is outside the timeslots.
        """

        for i, t in enumerate(self.timestamps):
            if (t.start_time <= timestamp) and (timestamp < t.end_time):
                return i
        return None

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

        for i, t in enumerate(self.__iter__()):
            start_slot.append(self._timestamp_to_timeslot(t.starttime))
            end_slot.append(self._timestamp_to_timeslot(t.endtime))

        self.trips["start_slot"] = start_slot
        self.trips["end_slot"] = end_slot

    def set_department_filter(self, departments):
        """
        Sets the department filter for the trips. Only trips belonging to the list of departments are included. If empty, all are included.

        Parameters
        ----------
        departments : list of department names as strings. Should match names from database.

        """
        # departments = np.atleast_1d(np.asanyarray(departments))
        if len(departments) == 0:
            self.department_filter = np.full((self.all_trips.shape[0],), True)
        else:
            self.department_filter = self.all_trips.isin(departments)

        self.set_filtered_trips()

    def set_date_filter(self, start=None, end=None):
        """
        Sets the department filter for the trips. Only trips belonging to the list of departments are included. If empty, all are included.

        Parameters
        ----------
        start : start date (included) as a pandas Timestamp. Default is None meaning open ended start date.
        end : end date (included) as a pandas Timestamp. Default is None meaning open ended end date.
        """

        print(f"Date_filter set to start={start} and end={end}")

        if start:
            filter_start = start <= self.all_trips.starttime
        else:
            filter_start = np.full((self.all_trips.shape[0],), True)

        if end:
            filter_end = self.all_trips.endtime <= end
        else:
            filter_end = np.full((self.all_trips.shape[0],), True)

        self.date_filter = filter_start & filter_end

        self.set_filtered_trips()

    def set_filtered_trips(self):
        """
        Applies date and department filter and sets model.trips accordingly.
        """
        # this function should create the filtered trips from filters
        self.trips = self.all_trips[self.date_filter & self.department_filter].copy()
        self.distance_range = (
            self.trips.distance.min(),
            self.trips.distance.max(),
        )

    def combine_subtrips(self, threshold=785e-8):
        """TODO

        Parameters
        ----------
        threshold : TODO
        """

        all_trips = self.all_trips
        round_trips = self.all_trips.iloc[0:0]
        for idd in all_trips.car_id.unique():
            df_sub = all_trips[all_trips.car_id == idd]
            phi0, lmbda0 = (
                df_sub.iloc[0][["start_latitude", "start_longitude"]] * np.pi / 180
            )
            phi, lmbda = (
                df_sub[["end_latitude", "end_longitude"]].values.T * np.pi / 180
            )
            d = np.arcsin(
                np.sqrt(
                    np.sin(0.5 * (phi - phi0)) ** 2
                    + np.cos(phi) * np.cos(phi0) * np.sin(0.5 * (lmbda - lmbda0)) ** 2
                )
            )
            cond = d < threshold
            cond[np.insert(~np.diff(cond), 0, cond[0])] = False
            indices = np.nonzero(cond)[0]
            total_dist = np.insert(df_sub.distance.values.cumsum()[indices], 0, 0)
            df_temp = df_sub.iloc[indices].copy()
            df_temp.distance = total_dist[1:] - total_dist[:-1]
            round_trips = round_trips.append(df_temp)
        self.all_trips = round_trips
        self.trips = round_trips


class ConsequenceCalculator:
    """ConsequenceCalculator class for computing economical, transport and emission consequences of a simulation

    Attributes
    ----------
    capacity_source : bokeh ColumnDataSource for containing capacity computed by the simulation
    consequence_table : bokeh ColumnDataSource for containing consequences computed by the simulation
    """

    def __init__(self):

        # consequence table
        self.table_keys = [
            "CO2-udledning [kg]",
            "NOx-udledning [g]",
            "Partikel-udledning [g]",
            "Årlige leasingomkostninger [kr/mnd]",
            "Antal ture uden køretøj",
            "Udbetalte kørepenge [kr]",
            "POGI Årlig gns. omkostning [kr/år]",
            "POGI CO2-ækvivalent udledning [CO2e]",
        ]
        self.sim_vals = [0, 0, 0, 0, 0, 0, 0, 0]
        self.cur_vals = [0, 0, 0, 0, 0, 0, 0, 0]
        self.consequence_table = ColumnDataSource()
        self.update_consequence_table()

        # capacity
        self.current_capacity = {"unassigned_trips": [0, 0], "trip_capacity": [0, 0]}
        self.simulation_capacity = {"unassigned_trips": [0, 0], "trip_capacity": [0, 0]}

        now = datetime.datetime.now()
        amonthback = now - datetime.timedelta(days=30)
        self.timeframe = [now, amonthback]
        self.capacity_source = ColumnDataSource()
        self.update_capacity_source()

    def update_consequence_table(self):
        """Update the consequence table"""
        self.consequence_table.data = {
            "keys": self.table_keys,
            "cur_values": self.cur_vals,
            "sim_values": self.sim_vals,
        }

    def update_capacity_source(self):
        """Update the capacity soruce"""
        d = {}
        d["timeframe"] = self.timeframe
        d["cur_unassigned_trips"] = self.current_capacity["unassigned_trips"]
        d["cur_trip_capacity"] = self.current_capacity["trip_capacity"]
        d["sim_unassigned_trips"] = self.simulation_capacity["unassigned_trips"]
        d["sim_trip_capacity"] = self.simulation_capacity["trip_capacity"]

        self.capacity_source.data = d

    def print_table(self):
        """Print the table, mostly for debugging"""
        print(self.consequence_table.data)

    def _compute_emissions(self, trips, key):
        """Compute emissions

        parameters
        ----------
        trips : trip data of type model.Trips
        key : name of simulation, typically 'current' or 'simulation'

        returns
        -------
        sum_co2 : sum of CO2 emissions for all trips
        sum_nox : sum of NOx emissions for all trips
        sum_particles : sum of particles emissions for all trips
        """

        # consequences on simulation
        co2 = []
        nox = []
        particles = []
        for t in trips:
            vehicle = t[key]
            distance = t["distance"]
            co2.append(distance * vehicle.co2emission_per_km)
            nox.append(distance * vehicle.NOx_per_km)
            particles.append(distance * vehicle.particles_per_km)

        trips.trips.loc[:, key[:3] + "_co2"] = co2
        trips.trips.loc[:, key[:3] + "_nox"] = nox
        trips.trips.loc[:, key[:3] + "_particles"] = particles
        return sum(co2), sum(nox), sum(particles)

    def _annual_average_distance_by_vehicletype(self, trips, col):
        """Annual average distance by vehicle type"""

        avg = dict()
        N = dict()
        days = (trips.endtime.max() - trips.starttime.min()).days

        for v in range(-1, 4):
            subset = trips[trips[col] == v]
            n = subset.shape[0]
            N[v] = n
            if n == 0:
                avg[v] = 0
            else:
                avg[v] = subset.distance.mean() / days * 365

        return avg, N, days

    def compute(self, simulation, drivingallowance):
        """Compute consequences of simulation

        parameters
        ----------
        simulation : simulation of type model.Simulation
        drivingallowance : model.DrivingAllowance object"""

        # consequences on simulation
        sim_co2, sim_nox, sim_particles = self._compute_emissions(
            simulation.trips, "simulation"
        )

        # consequences on current
        cur_co2, cur_nox, cur_particles = self._compute_emissions(
            simulation.trips, "current"
        )

        # unassigned vehicles
        sim_unas = (simulation.trips.trips["simulation_type"] == -1).sum()
        cur_unas = (simulation.trips.trips["current_type"] == -1).sum()

        # economy
        sim_tco = 0
        cur_tco = 0
        for v in simulation.fleet_manager.simulation_fleet:
            sim_tco = sim_tco + v.leasingydelse
        for v in simulation.fleet_manager.current_fleet:
            cur_tco = cur_tco + v.leasingydelse

        # driving allowance totals
        drivingallowance.update()
        cur_allowance, sim_allowance = drivingallowance.get_allowance()

        # compute capacity
        # for each day, compute number of trips
        ## update self.simulation_capacity and self.current_capacity
        sub = simulation.trips.trips[
            ["starttime", "current_type", "simulation_type"]
        ].copy(deep=True)
        sub["cur_unassigned"] = sub["current_type"] == -1
        sub["sim_unassigned"] = sub["simulation_type"] == -1
        resampled = sub.resample("D", on="starttime")[
            ["cur_unassigned", "sim_unassigned"]
        ].sum()

        self.timeframe = resampled.index.to_pydatetime()
        n = len(self.timeframe)
        self.current_capacity = {
            "unassigned_trips": list(resampled.cur_unassigned),
            "trip_capacity": n * [0],
        }
        self.simulation_capacity = {
            "unassigned_trips": list(resampled.sim_unassigned),
            "trip_capacity": n * [0],
        }

        # compute POGI/TCO
        # car = 0, ecar = 1
        cur_avg, cur_N, days = self._annual_average_distance_by_vehicletype(
            simulation.trips.trips, "current_type"
        )
        sim_avg, sim_N, days = self._annual_average_distance_by_vehicletype(
            simulation.trips.trips, "simulation_type"
        )

        # pogi TCO for car and e-car in current fleet
        pogi_cur_car = self._compute_pogi(
            cur_avg[0],
            simulation.fleet_manager.vehicle_factory.vmapper["Car"],
            cur_N[0],
        )
        pogi_cur_ecar = self._compute_pogi(
            cur_avg[1],
            simulation.fleet_manager.vehicle_factory.vmapper["ElectricCar"],
            cur_N[1],
        )
        cur_pogi_tco = pogi_cur_car[0] + pogi_cur_ecar[0]
        cur_pogi_ekstern = pogi_cur_car[1] + pogi_cur_ecar[1]

        # pogi TCO for car and e-car in simulation fleet
        pogi_sim_car = self._compute_pogi(
            sim_avg[0],
            simulation.fleet_manager.vehicle_factory.vmapper["Car"],
            sim_N[0],
        )
        pogi_sim_ecar = self._compute_pogi(
            sim_avg[1],
            simulation.fleet_manager.vehicle_factory.vmapper["ElectricCar"],
            sim_N[1],
        )
        sim_pogi_tco = pogi_sim_car[0] + pogi_sim_ecar[0]
        sim_pogi_ekstern = pogi_sim_car[1] + pogi_sim_ecar[1]

        # fill in values
        self.cur_vals = [
            cur_co2,
            cur_nox,
            cur_particles,
            cur_tco,
            cur_unas,
            cur_allowance,
            cur_pogi_tco,
            cur_pogi_ekstern,
        ]
        self.sim_vals = [
            sim_co2,
            sim_nox,
            sim_particles,
            sim_tco,
            sim_unas,
            sim_allowance,
            sim_pogi_tco,
            sim_pogi_ekstern,
        ]

        # update sources for frontend
        self.update_consequence_table()
        self.update_capacity_source()

    def _compute_pogi(self, distance, vehicle_type, n):
        """Compute POGI TCO from Excel sheet"""

        if vehicle_type.__name__ in ("Car", "ElectricCar"):
            pogi = PogiExcel()

            specs = dict(vehicle_type.__dict__)
            specs["koerselsforbrug"] = distance

            pogi = pogi.compute(specs)
            pogi = tuple([n * x for x in pogi])
        elif vehicle_type.__name__ in ("Bike", "ElectricBike"):
            pogi = (0, 0)
        else:
            raise ValueError("Arguemnt vehicle_type not specified correct.")

        return pogi

    def get_html_information(self):
        """Function for producing html used on infromation panel"""

        html = "<h2>Konsekvensberegning</h2>"
        html += "<table ><tr><th>Konsekvensmål</th><th>Nuværende værdi</th><th>Simuleret værdi</th></tr>"
        for t in zip(self.table_keys, self.cur_vals, self.sim_vals):
            html += f"<tr><td>{t[0]}</td><td>{t[1]:.02f}</td><td>{t[1]:.02f}</td></tr>"
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

    def __init__(self, options):

        self.options = options

        # set the available vehicles
        self.vehicle_factory = vehicle.VehicleFactory()
        self.vehicle_factory.load_options(options)

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

        html = "<h2>Flådeinformation</h2>"
        html += f"<dl><dt>Flådesammensætning, nuværende</dt><dd>Fossilbiler: {self.current_fleet.cars}</dd><dd>Elbiler: {self.current_fleet.ecars}</dd><dd>Cykler: {self.current_fleet.bikes}</dd><dd>Elcykler: {self.current_fleet.ebikes}</dd></dl>"
        html += f"<dl><dt>Flådesammensætning, simuleret</dt><dd>Fossilbiler: {self.simulation_fleet.cars}</dd><dd>Elbiler: {self.simulation_fleet.ecars}</dd><dd>Cykler: {self.simulation_fleet.bikes}</dd><dd>Elcykler: {self.simulation_fleet.ebikes}</dd></dl>"

        tmapper = {
            "Car": "Fossilbil",
            "ElectricCar": "Elbil",
            "Bike": "Cykel",
            "ElectricBike": "Elcykel",
        }
        for vtype in ("Car", "ElectricCar", "Bike", "ElectricBike"):
            html += f"<dl><dt>Køretøjstype: {tmapper[vtype]}:</dt>"

            vinfo = self.vehicle_factory.vmapper[vtype].__dict__
            for key, val in vinfo.items():
                if (key[:2] == "__") or (callable(val)):
                    continue
                html += f"<dd>{key}: {val}</dd>"
            html += "<dl/>"

        return html


class Simulation:
    """Simulation class for performing simulation on trips

    parameters
    ----------
    trips : trips for simulation of type modelTrips
    fleet_manager : fleet manager for handling fleets of type model.FleetManager

    """

    def __init__(self, trips, fleet_manager, progress_callback):
        self.trips = trips
        self.fleet_manager = fleet_manager
        self.progress_callback = progress_callback

        self.time_resolution = pd.Timedelta(minutes=60)
        start_day = self.trips.trips.starttime.min().date()
        end_day = self.trips.trips.endtime.max().date() + pd.Timedelta(days=1)
        self.timestamps = pd.period_range(start_day, end_day, freq=self.time_resolution)

        self.trips.set_timestamps(self.timestamps)

        # dummy vehicle for unassigned trips
        self.unassigned_vehicle = vehicle.Unassigned(name="Unassigned")

    def run(self):
        """Runs simulation of current and simulation fleet"""
        # push timetable to vehicle fleet
        self.fleet_manager.set_timestamps(self.timestamps)

        self.run_single(self.fleet_manager.simulation_fleet)
        self.run_single(self.fleet_manager.current_fleet)

    def run_single(self, fleet_inventory):
        """Convenience function for running simualtion on a single fleet

        parameters
        ----------
        fleet_inventory : fleet inventory to run simualtion on. Type model.FleetInventory.
        """
        # loop over trips
        trip_vehicle = []
        trip_vehicle_type = []

        for t in self.trips:
            # loop over vehicles and check for availability
            booked = False
            for v in fleet_inventory:
                booked, acc, avail = v.book_trip(t)

                if booked:
                    trip_vehicle.append(v)
                    trip_vehicle_type.append(v.vehicle_type_number)
                    # print(f"Tripid {t.tripid} assigned to vehicle {v.name}")
                    break

            if booked == False:
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
        self.current = {"low": 50, "high": 50}
        self.simulation = {"low": 50, "high": 50}
        self.distance = {"low": 300, "high": 100}
        self.allowance = {"low": 1.90, "high": 3.52}

        self.update()

    def __str__(self):
        return f"Driving allowance\n  Cur : {self.current}\n  Sim : {self.simulation}\n  Dist: {self.distance}\n  DKK : {self.allowance} "

    def set_distance(self, low=None, high=None):
        """Set distances for low and high allowance

        parameters
        ----------
        low : low allowance distance, km
        high : high allowance distance, km"""
        if low:
            self.distance["low"] = low

        if high:
            self.distance["high"] = high

        self.update()

    def set_allowance(self, low=None, high=None):
        """Set low and high allowance

        parameters
        ----------
        low : low allowance, kr/km
        high : high allowance, kr/ km
        """
        if low:
            self.allowance["low"] = low

        if high:
            self.allowance["high"] = high

        self.update()

    def set_options(self, opts):
        """Load options from OptionFile object

        parameters
        ----------
        opts : OptionFile.driwingallowance
        """

        low = opts[opts.Taksttype == "Lav"]
        high = opts[opts.Taksttype == "Høj"]

        self.set_distance(
            low=low["Antal km"].values[0], high=high["Antal km"].values[0]
        )
        self.set_allowance(low=low["Takst"].values[0], high=high["Takst"].values[0])

        self.__str__()

    def update(self):
        """Update driving allowance numbers"""
        # update
        self.total = self.distance["low"] + self.distance["high"]
        self.current["low"] = 100 * self.distance["low"] / self.total
        self.current["high"] = 100 * self.distance["high"] / self.total

    def get_allowance(self):
        """Get driving allownace

        returns
        -------
        cur : driving allowance for current distribution
        sim : driving allowance for simulation distribution
        """
        cur = self.current["low"] * self.total * self.allowance["low"]
        cur = cur + self.current["high"] * self.total * self.allowance["high"]
        cur = cur / 100  # from percent
        sim = self.simulation["low"] * self.total * self.allowance["low"]
        sim = sim + self.simulation["high"] * self.total * self.allowance["high"]
        sim = sim / 100  # from percent
        return cur, sim

    def get_html_information(self):
        """Function for producing html used on infromation panel"""

        html = "<h2>Kørepenge</h2>"
        html += f"<dl><dt>Kørselssatser</dt><dd>-Lav: {self.allowance['low']:.2f} kr/km</dd><dd>-Høj: {self.allowance['high']:.2f} kr/km</dd></dl>"
        html += f"<dl><dt>Nuværende fordeling af km på høj og lav sats</dt><dd>-Lav: {self.distance['low']:.1f} km ({self.current['low']:.1f}%)</dd><dd>-Høj: {self.distance['high']:.1f} km ({self.current['high']:.1f}%)</dd></dl>"
        total = self.distance["low"] + self.distance["high"]
        html += f"<dl><dt>Simuleret fordeling af km på høj og lav sats</dt><dd>-Lav: {total/100*self.simulation['low']:.1f} km ({self.simulation['low']:.1f}%)</dd><dd>-Høj: {total/100*self.simulation['high']:.1f} km ({self.simulation['high']:.1f}%)</dd></dl>"
        return html


class OptionFile:
    """Class for handling option file loading

    parameters
    ----------
    option_file : path to option file of type .xlsx. Default is the option file in sourcefiles/options.xlsx"""

    def __init__(self, option_file=None):

        if option_file:
            self.options_file = option_file
        else:
            # hardcoded for options file
            self.option_dir = pathlib.Path(__file__).parent.absolute() / "sourcefiles"
            self.options_file = self.option_dir / "options.xlsx"

        # load options in individual sheets
        self._load_vehicles()
        self._load_drivingallowances()
        self._load_pool()
        self._load_settings()

    def _load_vehicles(self):
        """Load vehicle types and selection from option file"""
        self.vehicletypes = pd.read_excel(
            self.options_file,
            sheet_name="Køretøjer",
            header=10,
            nrows=15,
            usecols="A:S",
        )

        self.vehicle_selection = pd.read_excel(
            self.options_file,
            sheet_name="Køretøjer",
            header=0,
            nrows=4,
            usecols="A:B",
        )

    def _load_drivingallowances(self):
        """Load driving allowances from option file"""
        self.drivingallowances = pd.read_excel(
            self.options_file,
            sheet_name="Befordringstakster",
            header=0,
            nrows=2,
            usecols="A:C",
        )

    def _load_pool(self):
        """Load vehicle pool settings from option file"""
        self.pool = pd.read_excel(
            self.options_file,
            sheet_name="Pulje",
            header=None,
            skiprows=1,
            nrows=2,
            usecols="A:B",
        )

    def _load_settings(self):
        """Load other settings from option file"""
        self.simulation_options = pd.read_excel(
            self.options_file,
            sheet_name="Simulering",
            nrows=2,
            usecols="A:B",
        )

    def __str__(self):
        return f"{self.vehicletypes}\n{self.vehicle_selection}"


class PogiExcel:
    """Class for handling computation of the POGI TCO via Excel. Refers to Excel documents in folder sourcefiles/pogi_xxx"""

    def __init__(self):
        self.version = "pogi_juni2021"
        self.financial_filename = "miljoestyrelsen-tco-vaerktoej-motorkoeretoejer-koeb-finansiel-leasing-2021.xlsm"
        self.operational_filename = "miljoestyrelsen-tco-vaerktoej-motorkoeretoejer-operationel-leasing-2021.xlsm"
        self.pogi_dir = pathlib.Path(__file__).parent / "sourcefiles" / self.version
        self.financial = self.pogi_dir / self.financial_filename
        self.operational = self.pogi_dir / self.operational_filename

        self.files_exist()

    def get_mapper(self, leasetype, drivmiddel):
        """Defines mapper functions for Excel files

        parameters
        ----------
        leasetype : one of ("finansiel", "operationel")
        drivmiddel: one of ("benzin", "diesel", "hybrid", "el", "plugin hybrid benzin", "plugin hybrid diesel")

        """

        sheet_mapper_op_bdh = {
            "etableringsgebyr": ("E12", 0),
            "braendstofforbrug": ("E15", 0),
            "leasingydelse": ("E18", 0),
            "ejerafgift": ("E19", 0),
            "antal": ("E39", 1),
            "koerselsforbrug": ("E40", 0),
            "evalueringsperiode": ("E46", 4),
            "drivmiddel": ("E49", "Benzin"),
            "forsikring": ("E64", 0),
            "loebende_omkostninger": ("E65", 0),
            "tco_pr_aar": ("E77", 0),
            "eksterne_virkninger": ("E79", 0),
        }

        sheet_mapper_op_eph = {
            "etableringsgebyr": ("E12", 0),
            "elforbrug": ("E15", 0),
            "leasingydelse": ("E18", 0),
            "ejerafgift": ("E19", 0),
            "braendstofforbrug": ("E24", 0),
            "antal": ("E48", 1),
            "koerselsforbrug": ("E49", 0),
            "evalueringsperiode": ("E55", 4),
            "opladningsabonnement": ("E63", 0),
            "drivmiddel": ("E65", "Plugin hybrid benzin"),
            "forsikring": ("E86", 0),
            "loebende_omkostninger": ("E87", 0),
            "tco_pr_aar": ("E100", 0),
            "eksterne_virkninger": ("E102", 0),
        }

        sheet_mapper_fin_bdh = {
            "indkobspris": ("E12", 0),
            "braendstofforbrug": ("E16", 0),
            "serviceaftale": ("E19", 0),
            "leasingydelse": ("E20", 0),
            "ejerafgift": ("E21", 0),
            "tilbagetagningspris": ("E24", 0),
            "antal": ("E44", 1),
            "koerselsforbrug": ("E45", 0),
            "evalueringsperiode": ("E51", 4),
            "drivmiddel": ("E54", "benzin"),
            "forsikring": ("E69", 0),
            "loebende_omkostninger": ("E70", 0),
            "tco_pr_aar": ("E84", 0),
            "eksterne_virkninger": ("E86", 0),
        }

        sheet_mapper_fin_eph = {
            "indkobspris": ("E12", 0),
            "elforbrug": ("E16", 0),
            "serviceaftale": ("E19", 0),
            "leasingydelse": ("E20", 0),
            "ejerafgift": ("E21", 0),
            "tilbagetagningspris": ("E24", 0),
            "braendstofforbrug": ("E29", 0),
            "antal": ("E53", 1),
            "koerselsforbrug": ("E54", 0),
            "evalueringsperiode": ("E60", 4),
            "opladningsabonnement": ("E68", 0),
            "drivmiddel": ("E70", "Plugin hybrid benzin"),
            "forsikring": ("E91", 0),
            "loebende_omkostninger": ("E92", 0),
            "tco_pr_aar": ("E106", 0),
            "eksterne_virkninger": ("E108", 0),
        }

        if leasetype == "operationel":
            if drivmiddel in ("benzin", "diesel", "hybrid"):
                return sheet_mapper_op_bdh
            if drivmiddel in ("el", "plugin hybrid benzin", "plugin hybrid diesel"):
                return sheet_mapper_op_eph
        if leasetype == "finansiel":
            if drivmiddel in ("benzin", "diesel", "hybrid"):
                return sheet_mapper_fin_bdh
            if drivmiddel in ("el", "plugin hybrid benzin", "plugin hybrid diesel"):
                return sheet_mapper_fin_eph

    def compute(self, specs, save_filename=None):
        """Compute POGI TCO via call to Excel

        parameters
        ----------
        save_filename : filename to save document to. Used for debugging. Defaults to None, ie. no saving"""

        # set mapper
        leasingtype = specs["leasingtype"].lower().strip()
        drivmiddel = specs["drivmiddel"].lower().strip()
        mapper = self.get_mapper(leasingtype, drivmiddel)

        # connect to app
        app = xw.App(visible=False)

        # connect to Excel sheet
        if leasingtype == "operationel":
            book = app.books.open(str(self.operational))
        elif leasingtype == "finansiel":
            book = app.books.open(str(self.financial))
        else:
            raise ValueError(
                "Argument 'leasingtype' must be either 'operationel' or 'finansiel'."
            )

        if drivmiddel in ("benzin", "diesel", "hybrid"):
            sheet = book.sheets["Benzin - diesel - hybrid"]
        elif drivmiddel in (
            "el",
            "plugin hybrid benzin",
            "plugin hybrid diesel",
        ):
            sheet = book.sheets["El og plugin hybrid"]
        else:
            raise ValueError(
                "Argument 'drivmiddel' must be either: benzin, diesel, hybrid, el, plugin hybrid benzin or plugin hybrid diesel"
            )

        for var, cell in mapper.items():
            if var in ("tco_pr_aar", "eksterne_virkninger"):
                continue

            if var in specs.keys():
                # take value from kwargs if present
                val = specs[var]
            else:
                # use defaults value if not in kwargs
                val = cell[1]

            sheet.range(cell[0]).value = val

        # get result
        tco = sheet.range(mapper["tco_pr_aar"][0]).value
        eksterne = sheet.range(mapper["eksterne_virkninger"][0]).value

        if save_filename:
            # book.save(save_filename)
            book.save(r"C:\Users\AllanLyckegaard\Downloads\tco_test_temp.xlsm")

        book.close()
        app.quit()

        return tco, eksterne

    def files_exist(self):
        """Test for file existence"""

        if not self.financial.is_file():
            raise OSError(f"{self.financial} is not a file.")

        if not self.operational.is_file():
            raise OSError(f"{self.operational} is not a file.")

    def __str__(self):
        return f"{self.financial}\n{self.operational}"


class Model:
    """Model class for MVC pattern of the simulation tool.

    Parameters
    ----------
    conn_info : connection info for database. Defauts to None which loads a dummy dataset

    """

    def __init__(self, conn_info=None):

        self.options = OptionFile()

        if conn_info is None:
            self.trips = Trips("dummy")
        else:
            self.trips = Trips(data_from_db(conn_info))

        self.trips.set_date_filter(
            self.options.pool.iloc[0, 1], self.options.pool.iloc[1, 1]
        )

        self.fleet_manager = FleetManager(self.options)

        self.consequence_calculator = ConsequenceCalculator()

        # static references to data sources needed by the view
        self.consequence_source = self.consequence_calculator.consequence_table
        self.capacity_source = self.consequence_calculator.capacity_source
        self.progress_source = ColumnDataSource(
            data={"start": [0.0], "progress": [0.0]}
        )
        self.progress_callback = lambda x: print(f"Simulér ({100*x}%)")

        # update histogram sources
        self.current_hist_datasource = ColumnDataSource()
        self.simulation_hist_datasource = ColumnDataSource()
        self.compute_histogram()

        # driving allowance
        self.drivingallowance = DrivingAllowance()
        self.drivingallowance.set_options(self.options.drivingallowances)

    def _update_progress(self, progress):
        """Tester function for updating progress of simualtion"""
        self.progress_source.data = {"start": [0.0], "progress": [progress]}
        if progress > (1.0 - 1e-12):
            self.progress_callback(False)
        else:
            self.progress_callback(True)

    def run_simulation(self):
        """Create and run a simulation. Updates histograms and consequence information."""

        # init simulation
        self.simulation = Simulation(
            self.trips, self.fleet_manager, self._update_progress
        )

        # collect data from frontend
        self.simulation.fleet_manager.current_fleet.initialise_fleet()
        self.simulation.fleet_manager.simulation_fleet.initialise_fleet()

        self.simulation.run()

        # update data sources for frontend
        self.compute_histogram()

        # update consequence sources for frontend
        self.consequence_calculator.compute(self.simulation, self.drivingallowance)

    # def get_trips_histogram(self):
    #    return self.trips.trips_histograms

    def load_data_file(self, attr, old, f):
        # f is a str with bytes
        dept = pd.read_excel(
            base64.b64decode(f),
            sheet_name="Pulje",
            header=None,
            nrows=1,
            usecols="B",
        ).values[0, 0]
        self.trips.set_department_filter(dept)

    def compute_histogram(self, mindist=0, maxdist=None):
        """Compute histograms for current and simulation

        parameters
        ----------
        mindist : defaults to 0. Minimum distance to use for histograms
        maxdist : Maximum distance to use for histograms. If None, use the maximum distance of the trips

        """

        if maxdist is None:
            maxdist = self.trips.distance_range[1]

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
