from operator import attrgetter

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (
    Button,
    ColumnDataSource,
    DataTable,
    DatetimeTickFormatter,
    Div,
    FileInput,
    HoverTool,
    MultiSelect,
    NumericInput,
    Spacer,
    TableColumn,
    RangeSlider,
)
from bokeh.models.widgets import NumberFormatter, Panel, Tabs
from bokeh.plotting import figure


class View:
    """View class for MVC pattern of the simulation tool.

    Parameters
    ----------
    model : model of type model.Model

    """

    def __init__(self, model):
        self.model = model

        # supporting tools
        self.round2_formatter = NumberFormatter(format="0.00", text_align="right")

        # define the overall view
        self.inputs = column(
            *self.view_fleet_input(),
            self.view_departments(),
            self.view_drivingallowance(),
            # self.view_button_fileinput(),
            self.view_button_simulate()
        )
        self.middle = column(*self.view_trips_plot())
        self.computations = column(
            *self.view_consequence_table(), *self.view_capacity()
        )
        self.spacer_left = Spacer(width=50)
        self.spacer_right = Spacer(width=50)

        self.model_info = Div(text="", width=800)

        self.gui = Tabs(
            tabs=[
                Panel(
                    child=row(
                        self.inputs,
                        self.spacer_left,
                        self.middle,
                        self.spacer_right,
                        self.computations,
                        width=1600,
                    ),
                    title="Simulering",
                ),
                Panel(child=self.model_info, title="Information"),
            ]
        )

    def view_consequence_table(self):
        """Define view for consequence table"""

        columns = [
            TableColumn(field="keys", title="Konsekvensmål"),
            TableColumn(
                field="cur_values", title="Nuv. værdi", formatter=self.round2_formatter
            ),
            TableColumn(
                field="sim_values", title="Sim. værdi", formatter=self.round2_formatter
            ),
        ]

        prv = Div(text="<h2>Konsekvensberegning</h2>")
        consequences = DataTable(
            source=self.model.consequence_source,
            columns=columns,
            width=400,
            height=300,
            index_position=None,
        )

        # prv0 = Div(text="<h2>Kapacitet i systemet</h2>")
        # prv1 = Div(text=f"Andel af ture som ikke dækkes af flåden: {0.3} %<br>Andel ture med i privalbil: {23.7} %")

        return prv, consequences  # , prv0, prv1

    def capacity_reset_range(self):
        """Reset capacity range from model"""

        # capacity_source = self.model.capacity_source
        mindate = min(self.model.capacity_source.data["timeframe"])
        maxdate = max(self.model.capacity_source.data["timeframe"])

        # convert to ms after epoch 1-1-1970
        minval = mindate.timestamp() * 1000
        maxval = maxdate.timestamp() * 1000

        self.cap.x_range.start = minval
        self.cap.x_range.end = maxval

    def view_capacity(self):
        """Define view for capacity"""
        # set source
        capacity_source = self.model.capacity_source

        ttip = HoverTool(
            tooltips=[
                ("Tidspunkt", "@timeframe{%F}"),
                ("Ikke tildelte, nuv.", "@cur_unassigned_trips"),
                ("Ikke tildelte, sim.", "@sim_unassigned_trips"),
            ],
            formatters={
                "@timeframe": "datetime",
                # use 'datetime' formatter
                # use default 'numeral' formatter for other fields
            },
            # display a tooltip whenever the cursor is vertically in line with a glyph
            mode="vline",
        )

        heading = Div(text="<h2>Kapacitet</h2>")
        cap = figure(
            plot_width=600,
            plot_height=400,
            x_axis_type="datetime",
        )

        cap.add_tools(ttip)
        cap.toolbar.active_drag = None
        cap.toolbar.active_scroll = None
        cap.toolbar.active_tap = None

        cap.xaxis.axis_label = "Tidspunkt"
        cap.yaxis.axis_label = "Antal ture uden køretøj"

        cap.xaxis.formatter = DatetimeTickFormatter(
            days=["%Y-%m-%d"], months=["%Y-%m-%d"], years=["%Y-%m-%d"]
        )
        cap.xaxis.major_label_orientation = np.pi / 3

        # plot lines for unassigned trips
        cap.line(
            x="timeframe",
            y="cur_unassigned_trips",
            line_width=2,
            color="blue",
            legend_label="Current",
            source=capacity_source,
        )
        cap.line(
            x="timeframe",
            y="sim_unassigned_trips",
            line_width=2,
            color="green",
            legend_label="Simulation",
            source=capacity_source,
        )

        # capacity lines
        # cap.line([-30, 0], [100, 100], line_width=2, color="red")
        # cap.line([-30, 0], [85, 85], line_width=2, color="red")

        self.cap = cap

        # self.capacity_reset_range()
        return heading, cap

    def view_trips_plot(self):
        """Define view for trip histograms"""

        maxdist = self.model.trips.distance_range[1]
        delta = maxdist / 20 * 0.9

        # set source
        source_current = self.model.current_hist_datasource

        p2 = figure(
            title="Nuværende fordeling af kørte km", plot_height=400, plot_width=500
        )
        # distances = sourcedrive
        types = list(source_current.data.keys())
        types.remove("edges")
        p2.vbar_stack(
            types,
            x="edges",
            source=source_current,
            width=delta,
            color=["gray", "red", "blue", "green", "orange"],
            legend_label=types,
            name="hist_cur",
        )
        p2.legend.location = "top_right"
        p2.legend.orientation = "horizontal"
        p2.xaxis.axis_label = "Turlængde [km]"

        p2.toolbar.active_drag = None
        p2.toolbar.active_scroll = None
        p2.toolbar.active_tap = None

        p2a = figure(
            title="Simuleret fordeling af kørte km", plot_height=400, plot_width=500
        )
        # distances = sourcedrive
        source_simulation = self.model.simulation_hist_datasource

        types = list(source_simulation.data.keys())
        types.remove("edges")
        p2a.vbar_stack(
            types,
            x="edges",
            source=source_simulation,
            width=delta,
            color=["gray", "red", "blue", "green", "orange"],
            legend_label=types,
            name="hist_sim",
        )
        p2a.legend.location = "top_right"
        p2a.legend.orientation = "horizontal"
        p2a.xaxis.axis_label = "Turlængde [km]"

        p2a.toolbar.active_drag = None
        p2a.toolbar.active_scroll = None
        p2a.toolbar.active_tap = None

        # trip min,max adjustment
        maxval = int(self.model.trips.distance_range[1] + 0.5)
        self.trip_range = RangeSlider(
            start=0,
            end=maxval,
            value=(0, maxval),
            step=1,
            title="Afstandsvælger [km]",
        )

        self.hist_cur = p2
        self.hist_sim = p2a
        return self.hist_cur, self.hist_sim, self.trip_range

    def view_drivingallowance(self):
        """Define view for driving allowance"""
        # kørepenge policy
        heading = Div(text="<h2>Kørsel i privatbil</h2>")
        # kp_cp = CheckboxGroup(labels = ['Inkluder data for privat kørsel'])
        h_cur = Div(text="<h3>Nuværende fordeling kørepengesats</h3>")
        self.da_cur_low = NumericInput(
            title="Lav",
            low=0,
            high=100,
            width=70,
            disabled=True,
            format="0.0",
            mode="float",
        )
        self.da_cur_high = NumericInput(
            title="Høj",
            low=0,
            high=100,
            width=70,
            disabled=True,
            format="0.0",
            mode="float",
        )

        h_sim = Div(text="<h3>Simuleret fordeling kørepengesats</h3>")
        self.da_sim_low = NumericInput(
            title="Lav", low=0, high=100, width=70, format="0.0", mode="float"
        )
        self.da_sim_high = NumericInput(
            title="Høj", low=0, high=100, width=70, format="0.0", mode="float"
        )

        da = column(
            heading,
            h_cur,
            row(self.da_cur_low, self.da_cur_high),
            h_sim,
            row(self.da_sim_low, self.da_sim_high),
        )
        return da

    def view_fleet_input(self):
        """Define view for input to fleet"""
        # current fleet input
        cur_heading = Div(text="<h2>Nuværende flådesammensætning</h2>")
        self.cur_car = NumericInput(title="Fossilbil", low=0, width=50)
        self.cur_ecar = NumericInput(title="Elbil", low=0, width=50)
        self.cur_bike = NumericInput(title="Cykel", low=0, width=50)
        self.cur_ebike = NumericInput(title="Elcykel", low=0, width=50)

        nu = column(
            cur_heading, row(self.cur_car, self.cur_ecar, self.cur_bike, self.cur_ebike)
        )

        # simulation fleet input
        sim_heading = Div(text="<h2>Simuleret flådesammensætning</h2>")
        self.sim_car = NumericInput(title="Fossilbil", low=0, width=50)
        self.sim_ecar = NumericInput(title="Elbil", low=0, width=50)
        self.sim_bike = NumericInput(title="Cykel", low=0, width=50)
        self.sim_ebike = NumericInput(title="Elcykel", low=0, width=50)

        sim = column(
            sim_heading, row(self.sim_car, self.sim_ecar, self.sim_bike, self.sim_ebike)
        )

        return nu, sim

    # def view_button_fileinput(self):
    #    """Define view for consequence table"""
    #    self.file_heading = Div(text="<h2>Upload af baggrundsdata</h2>")
    #    self.file_input = FileInput(accept=".xlsx")
    #    return column(self.file_heading, self.file_input)

    def view_button_simulate(self):
        """Define view for simulate button"""

        self.simulate_heading = Div(text="<h2>Simulering</h2>")
        self.simulate_button = Button(label="Simulér")
        self.sim_progress = figure(
            title="Simulering, progress", plot_height=100, plot_width=300
        )

        # progress bar for simulation
        self.sim_progress.visible = False
        self.sim_progress.yaxis.visible = False
        self.sim_progress.x_range.start = 0.0
        self.sim_progress.x_range.end = 1.0
        self.sim_progress.hbar(
            y=0,
            left="start",
            right="progress",
            color="green",
            source=self.model.progress_source,
        )

        return column(self.simulate_heading, self.simulate_button, self.sim_progress)

    def view_departments(self):
        """Define view for department selector"""

        self.dept_heading = Div(text="<h2>Pulje</h2>")
        depts = self.model.trips.all_trips["department"].unique().tolist()
        self.dept_select = MultiSelect(value=depts, options=depts)
        return column(self.dept_heading, self.dept_select)

    def view_info_table(self, model_prop, table=None):
        """Define view for information table"""
        data = {}
        columns = []
        for label, prop in model_prop.items():
            data[prop] = np.atleast_1d(attrgetter(prop)(self.model))
            columns.append(TableColumn(field=prop, title=label))
        source = ColumnDataSource(data=data)
        if table:
            table.source = source
            table.columns = columns
        else:
            return DataTable(source=source, columns=columns)

    def set_progress(self):
        """Helper function to set progress. Currently not used."""
        p = self.model.progress_source.data["progress"]
        print("Progress called:", p)
        if (p == 0) or (p == 1):
            self.sim_progress.visible = False
        else:
            self.sim_progress.visible = True

    def update_infotab(self):
        """Helper function to update info panel."""
        # loop over model objects and get information tabels
        objs = [
            self.model.fleet_manager,
            self.model.consequence_calculator,
            self.model.drivingallowance,
        ]
        self.model_info.text = "<h1>Information om simulering</h1>"
        for obj in objs:
            self.model_info.text += "<br><hl><br>" + obj.get_html_information()
