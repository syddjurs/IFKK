import model
import view


class Controller:
    def __init__(self, curdoc):
        """Controller class for MVC pattern of the simulation tool.

        Parameters
        ----------
        curdoc : bokeh's document for the current default state.

        """

        # init model
        self.model = model.Model()
        # self.model.progress_callback = self.toggle_progress

        # init view
        self.view = view.View(self.model)

        # update view from model
        self.update_view_from_model()

        # hook up all callbacks
        self.set_callbacks()

        # generate view on root
        curdoc().add_root(self.view.gui)
        # curdoc().add_periodic_callback(self.view.set_progress, 1000)

    def set_callbacks(self):
        """Set all callbacks for the view."""

        # add callbacks to view for current fleet
        self.view.cur_car.on_change("value", self.update_current_car)
        self.view.cur_ecar.on_change("value", self.update_current_ecar)
        self.view.cur_bike.on_change("value", self.update_current_bike)
        self.view.cur_ebike.on_change("value", self.update_current_ebike)

        # add callbacks to view for current fleet
        self.view.sim_car.on_change("value", self.update_simulation_car)
        self.view.sim_ecar.on_change("value", self.update_simulation_ecar)
        self.view.sim_bike.on_change("value", self.update_simulation_bike)
        self.view.sim_ebike.on_change("value", self.update_simulation_ebike)

        # add callbacks to view for driving allowance
        self.view.da_cur_low.on_change("value", self.update_current_da_low)
        self.view.da_cur_high.on_change("value", self.update_current_da_high)
        self.view.da_sim_low.on_change("value", self.update_simulation_da_low)
        self.view.da_sim_high.on_change("value", self.update_simulation_da_high)

        # add callback to view for trip range
        self.view.trip_range.on_change("value", self.update_trip_range)

        # add callback to run simulation
        self.view.simulate_button.on_click(self.run_simulation)

        # add callback to apply choice of department
        self.view.dept_select.on_change("value", self.update_departments)

    def run_simulation(self):
        """
        Run simulation.
        """
        self.model.run_simulation()
        self.view.capacity_reset_range()
        self.view.update_infotab()

        # self.view.dept_select.on_change("value", self.update_departments)

    def update_current_da_low(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.drivingallowance.current["low"] = new
        self.model.drivingallowance.current["high"] = 100 - new
        self.update_view_from_model()

    def update_current_da_high(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.drivingallowance.current["low"] = 100 - new
        self.model.drivingallowance.current["high"] = new
        self.update_view_from_model()

    def update_simulation_da_low(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.drivingallowance.simulation["low"] = new
        self.model.drivingallowance.simulation["high"] = 100 - new
        self.model.drivingallowance.update()
        self.update_view_from_model()

    def update_simulation_da_high(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.drivingallowance.simulation["low"] = 100 - new
        self.model.drivingallowance.simulation["high"] = new
        self.model.drivingallowance.update()
        self.update_view_from_model()

    def update_current_bike(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.fleet_manager.current_fleet.bikes = new

    def update_current_ebike(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.fleet_manager.current_fleet.ebikes = new

    def update_current_car(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.fleet_manager.current_fleet.cars = new

    def update_current_ecar(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.fleet_manager.current_fleet.ecars = new

    def update_simulation_bike(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.fleet_manager.simulation_fleet.bikes = new

    def update_simulation_ebike(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.fleet_manager.simulation_fleet.ebikes = new

    def update_simulation_car(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.fleet_manager.simulation_fleet.cars = new

    def update_simulation_ecar(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.fleet_manager.simulation_fleet.ecars = new

    def update_departments(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        try:
            self.model.trips.set_department_filter(new)
        except KeyError:
            pass

    def update_trip_range(self, attr, old, new):
        """Callback funtion for updating the model from the view"""
        self.model.compute_histogram(mindist=new[0], maxdist=new[1])
        delta = (new[1] - new[0]) / 20.0
        h = self.view.hist_cur.select(name="hist_cur")
        for g in h:
            h.glyph.width = delta * 0.9
        h = self.view.hist_sim.select(name="hist_sim")
        for g in h:
            h.glyph.width = delta * 0.9

    def update_view_from_model(self):
        """Update the view from the model"""
        # populate view with values from model
        self.view.cur_car.value = self.model.fleet_manager.current_fleet.cars
        self.view.cur_ecar.value = self.model.fleet_manager.current_fleet.ecars
        self.view.cur_bike.value = self.model.fleet_manager.current_fleet.bikes
        self.view.cur_ebike.value = self.model.fleet_manager.current_fleet.ebikes

        self.view.sim_car.value = self.model.fleet_manager.simulation_fleet.cars
        self.view.sim_ecar.value = self.model.fleet_manager.simulation_fleet.ecars
        self.view.sim_bike.value = self.model.fleet_manager.simulation_fleet.bikes
        self.view.sim_ebike.value = self.model.fleet_manager.simulation_fleet.ebikes

        self.view.da_cur_low.value = self.model.drivingallowance.current["low"]
        self.view.da_cur_high.value = self.model.drivingallowance.current["high"]

        self.view.da_sim_low.value = self.model.drivingallowance.simulation["low"]
        self.view.da_sim_high.value = self.model.drivingallowance.simulation["high"]

        # self.view.trip_range.start = floorself.model.trips.distance_range[0].floor()
        self.view.trip_range.end = int(self.model.trips.distance_range[1] + 0.5)
