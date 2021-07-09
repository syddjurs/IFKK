import numpy as np

# initialise global mappers
vehicle_mapping = {-1: "Ikke tildelt", 0: "Bil", 1: "Elbil", 2: "Cykel", 3: "Elcykel"}


class VehicleModel:
    """General vehicle model. Not directly instantiated but specific vehicle models inherit from it"""

    # emissions
    co2emission_per_km = None  # mg/km
    NOx_per_km = None  # mg/km
    particles_per_km = None  # mg/km
    max_distance_per_day = None
    # economy
    tco_per_month = None

    # max_speed: float = 80
    # _timestamps = []

    def __init__(self, name=None, vehicle_id=None):
        self.vehicle_id = vehicle_id
        self.timeslots = []
        self.name = name

    def set_timestamps(self, new_timestamps):
        """Initailize timeslot of vehicle to match timestamps.

        parameters
        ----------
        new_timestamp : array of timestamps
        """
        self.timestamps = new_timestamps

        # initialise timeslots here
        self.timeslots = np.zeros((len(self.timestamps),), dtype=int)

    def book_trip(self, trip):
        """Function for trying to book trip on vehicle

        parameters
        ----------
        trip : pandas row with a trip

        returns
        -------
        booked : boolean, True if vehicle is booked on this vehile
        accept : boolean, True if it can accept the type of trip
        available: boolean, True if the vehicle is available in this timeperiod
        """
        # try to book on this vehicle. Returns a tuple with (accept, available). Return value 'accept' is true if the vehicle can perform this trip and 'available' is true if it is available in this

        # test acceptance of trip
        accept = self.accept_trip(trip)

        try:
            start_slot = int(trip.start_slot)
        except:
            start_slot = np.NaN

        try:
            end_slot = int(trip.end_slot)
        except:
            end_slot = np.NaN

        # print(trip)
        if (start_slot is np.NaN) or (end_slot is np.NaN):
            print(f"start_slot={start_slot}, end_slot={end_slot}")
            return False, False, False

        # lookup in timeslots list
        try:
            if any(self.timeslots[start_slot : (end_slot + 1)] > 0):
                available = False
            else:
                available = True
        except:
            print(start_slot)
            print(end_slot)
            available = False

        # collect and book
        if accept and available:
            # vehicle accept trip and vehicle available
            self.timeslots[start_slot : (end_slot + 1)] = trip.tripid
            return True, accept, available
        else:
            # vehicle not available or cannot accept trip
            return False, accept, available

    def accept_trip(self, trip):
        """function that returns true if the given type of trip is possible for the vehicle given length, duration, milage_left should be overwritten for each type of vehicle

        parameters
        ----------
        trip : pandas row with a trip
        """
        return True

    def __str__(self):
        return f"{self.__class__.__name__}(id={self.vehicle_id}, name={self.name})"

    @classmethod
    def class_info(self):
        return f"(co2emission_per_km={self.co2emission_per_km}, NOx_per_km={self.NOx_per_km}, particles_per_km={self.particles_per_km}, max_distance_per_day={self.max_distance_per_day}, tco_per_month={self.tco_per_month})"

    def print_info(self):
        """Print info. Mostly for debugging"""
        print(self.name)
        print(self.timestamps)
        print(self.timeslots)


class Car(VehicleModel):
    """Class for representing a car"""

    vehicle_type_number = 0
    max_distance_per_day = 9999
    co2emission_per_km = 150
    NOx_per_km = 3  # mg/km
    particles_per_km = 4  # mg/km
    tco_per_month = 1800.33  # DKK/month

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ElectricCar(VehicleModel):
    """Class for representing an electric car"""

    vehicle_type_number = 1
    max_distance_per_day = 200
    co2emission_per_km = 48
    NOx_per_km = 0  # mg/km
    particles_per_km = 0  # mg/km
    tco_per_month = 2759.26  # DKK/month

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # simple battery indicator
        self.milage_left = self.max_distance_per_day

    def accept_trip(self, trip):
        # accept if enough km left on car and update milage_left
        if trip.distance < self.milage_left:
            # self.milage_left = self.milage_left - trip.distance
            return True
        else:
            # trip is too long to accept
            return False


class Bike(VehicleModel):
    """Class for representing a bike"""

    vehicle_type_number = 2
    max_distance_per_day = 20
    co2emission_per_km = 5
    NOx_per_km = 0  # mg/km
    particles_per_km = 0  # mg/km
    tco_per_month = 175.0  # DKK/month

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def accept_trip(self, trip):
        # accept if enough km left on car
        if trip.distance < self.max_distance_per_day:
            return True
        else:
            # trip is too long to accept
            return False


class ElectricBike(VehicleModel):
    """Class for representing an electric bike"""

    vehicle_type_number = 3
    max_distance_per_day = 50
    co2emission_per_km = 20
    NOx_per_km = 0  # mg/km
    particles_per_km = 0  # mg/km
    tco_per_month = 600.20  # DKK/month

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def accept_trip(self, trip):
        # accept if enough km left on car
        if trip.distance < self.max_distance_per_day:
            return True
        else:
            # trip is too long to accept
            return False


class Unassigned(VehicleModel):
    """Class for representing an unassigned vehicle. Used for computing capacity."""

    vehicle_type_number = -1
    max_distance_per_day = 0
    co2emission_per_km = 0
    NOx_per_km = 0  # mg/km
    particles_per_km = 0  # mg/km
    tco_per_month = 0  # DKK/month

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def accept_trip(self, trip):
        # always accept
        return True


class VehicleFactory:
    """Class for containing the vehicle types that are used in the simulation"""

    def __init__(self):
        # type to class mapper
        self.vmapper = {
            "Bike": Bike,
            "ElectricBike": ElectricBike,
            "Car": Car,
            "ElectricCar": ElectricCar,
            "Unassigned": Unassigned,
        }

        self.columnmapper = {
            "Model": "model",
            "Drivmiddel": "drivmiddel",
            "Rækkevidde [km]": "max_distance_per_day",
            "CO2-udledning [g/km]": "co2emission_per_km",
            "NOx-udledning [mg/km]": "NOx_per_km",
            "Partikel-udledning [mg/km]": "particles_per_km",
            "Brændstofforbrug WLTP [km/L]": "braendstofforbrug",
            "Elforbrug WLTP [Wh/km]": "elforbrug",
            "Leasingtype": "leasingtype",
            "Etableringsgebyr [kr]": "etableringsgebyr",
            "Indkøbspris [kr]": "indkobspris",
            "Serviceaftale [kr/år]": "serviceaftale",
            "Leasingydelse [kr/år]": "leasingydelse",
            "Grøn ejerafgift [kr/år]": "ejerafgift",
            "Tilbagetagningspris [kr]": "tilbagetagningspris",
            "Forsikring [kr/år]": "forsikring",
            "Andre løbende omkostninger  [kr/år]": "loebende_omkostninger",
            "Note": "note",
        }

    def get_new_vehicle(self, vtype, name, vehicle_id):
        """Function for creating new vehicles

        parameters
        ----------
        vtype : string, vtype as key into vmapper, one of ("Bike", "ElectricBike", "Car", "ElectricCar","Unassigned")
        name : string, name of vehicle
        vehicle_id : int, id of vehicle
        returns
        -------
        """
        return self.vmapper[vtype](name, vehicle_id)

    def set_vehicle_options(self, vtype, df):
        """Helper function to set vehicle options"""
        for key, data in df.iteritems():
            mkey = self.columnmapper[key]
            setattr(self.vmapper[vtype], mkey, data.values[0])

    def load_options(self, options):
        """Load options from option file"""

        mapper = {
            "Fossilbil": "Car",
            "Elbil": "ElectricCar",
            "Elcykel": "ElectricBike",
            "Cykel": "Bike",
        }

        # loop over selection and push settings to vehicle types
        for label, vtype in mapper.items():
            choice = options.vehicle_selection.loc[
                options.vehicle_selection["Køretøjstype"] == label
            ]

            specs = options.vehicletypes[
                options.vehicletypes["Model"] == choice.Model.values[0]
            ]
            specs = specs.drop("Køretøjstype", axis=1)

            self.set_vehicle_options(vtype, specs)

    def __str__(self):
        out = ""
        for key, val in self.vmapper.items():
            out = out + f"{val.__name__}{val.class_info()}\n"
        return out


class FleetInventory:
    """Class for containing a fleet of vehicles.

    parameters
    ----------
    models: available models type model.VehicleFactory
    bikes : int, number of bikes in fleet
    ebikes: int, number of electric bikes in fleet
    cars: int, number of cars in fleet
    ecars: int, number of electric cars in fleet
    name: string, name of the fleet, typically 'simulation' or 'current'
    """

    def __init__(
        self, models, bikes=0, ebikes=0, cars=0, ecars=0, name="fleetinventory"
    ):
        self.bikes = bikes
        self.ebikes = ebikes
        self.cars = cars
        self.ecars = ecars

        self.models = models

        self.name = name

        self.initialise_fleet()

    def initialise_fleet(self):
        """Initialise fleet"""
        # clear list of vehicles
        self.vehicles = []

        # init vehicles in order of their (environmental) priority
        for t in zip(
            (self.bikes, self.ebikes, self.ecars, self.cars),
            ("Bike", "ElectricBike", "ElectricCar", "Car"),
        ):
            n, c = t
            for i in range(n):
                v = self.models.get_new_vehicle(
                    vtype=c, name=f"{c}_{i+1}", vehicle_id=self.get_total()
                )
                self.vehicles.append(v)
                # print(f"Adding {v}")

    def print_timetable(self):
        """Print timetable for fleet. Mostly for debugging."""

        print(f"Timetable for fleet: {self.name}")
        print(f"{'Timestamp':>20}", end="")
        for v in self:
            print(f"{v.name[:2]+v.name[-2:]:>6}", end="")
        print("")

        for i, t in enumerate(self.timestamps):
            print(f"{str(t):>20}", end="")

            for v in self:
                tripid = v.timeslots[i]
                if tripid == 0:
                    tripid = "."
                print(f"{tripid:>6}", end="")
            print("")

    def get_total(self):
        return len(self.vehicles)

    def set_timestamps(self, timestamps):
        """set timestamps for all vehicles"""
        self.timestamps = timestamps

        # loop over all vehicles and set the timestamps
        for v in self.__iter__():
            v.set_timestamps(self.timestamps)

    def __str__(self):
        out = f"Fleetinventory: {self.name}\n"
        out = out + f" Bikes  : {self.bikes}\n"
        out = out + f" E-bikes: {self.ebikes}\n"
        out = out + f" Cars:  : {self.cars}\n"
        out = out + f" E-cars : {self.ecars}\n"
        out = out + 12 * "-" + "\n"
        out = out + f" Total  : {self.get_total()}\n"
        return out

    def __iter__(self):
        # iterator that first traverses bikes, ebikes, ecars and then cars
        yield from self.vehicles
