import model
import vehicle


def example1():

    vehicle_models = vehicle.VehicleFactory()

    fleet_manager = model.FleetManager(vehicle_models)
    fleet_manager.current_fleet.bikes = 3
    fleet_manager.current_fleet.cars = 1
    fleet_manager.current_fleet.initialise_fleet()

    fleet_manager.simulation_fleet.bikes = 3
    fleet_manager.simulation_fleet.cars = 2
    fleet_manager.simulation_fleet.ecars = 2
    fleet_manager.simulation_fleet.initialise_fleet()

    trips = model.Trips(dataset="dummy_outside")
    s = model.Simulation(trips, fleet_manager)
    s.run()

    cons = model.ConsequenceCalculator()
    cons.compute(s)

    cons.update_consequence_table()
    cons.print_table()

    print(trips.trips)


def example2():
    m = model.Model()

    m.fleet_manager.set_current_fleet(bikes=1, ebikes=1, ecars=0, cars=3)
    m.fleet_manager.set_simulation_fleet(bikes=1, ebikes=1, ecars=0, cars=3)

    m.run_simulation()

    m.consequence_calculator.print_table()

    m.run_simulation()

    m.consequence_calculator.print_table()

    print(m)


def example_optionsfile():
    opt = model.Model()


def example4():
    trip = model.Trips(dataset="dummy").trips.iloc[0]

    m = model.Model()

    print(m.fleet_manager.vehicle_factory)

    bike1 = m.fleet_manager.vehicle_factory.get_new_vehicle("Bike", "bike1", 1)
    print(bike1.max_distance_per_day)

    m.fleet_manager.vehicle_factory.set_vehicle_options(
        "Bike", max_distance_per_day=99999
    )

    print()
    print(m.fleet_manager.vehicle_factory)

    bike2 = m.fleet_manager.vehicle_factory.get_new_vehicle("Bike", "bike2", 2)
    print(bike2.max_distance_per_day)


def example5():

    m = model.PogiExcel()
    print(m)
    tco, eks = m.compute(
        "operational",
        "Benzin",
        etableringsgebyr=100000,
        braendstofforbrug=20,
        pris_via_leasing=25000,
        ejerafgift=500,
        koerselsforbrug=15000,
        forsikring=5000,
        loebende_omkostninger=1000,
    )
    print(tco, eks)


if __name__ == "__main__":
    # example2()
    example_optionsfile()
