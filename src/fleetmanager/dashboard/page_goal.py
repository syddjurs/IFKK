import datetime
import pdb

import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import dash_daq as daq
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, MATCH, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from sqlalchemy import func, or_, and_
from sqlalchemy.orm import Query

import fleetmanager.data_access.db_engine as db
from ..data_access import Cars
from .utils import (
    accordian_table,
    card,
    get_emission,
    goal_bar_plot,
    unix_time_millis,
    unix_to_datetime,
)
from fleetmanager.model.tabu import TabuSearch
from fleetmanager.model.vehicle_optimisation import FleetOptimisation

fig = go.Figure()

from .view import app

Session = db.session_factory(db.engine_creator())

description_header = dbc.Toast(
    [
        html.P(
            "På denne side kan man som bruger anmode AI modulet om at komme med forslag til nye flådesammensætninger",
            className="mb-0",
        )
    ],
    header="Flådesammensætning",
    className="description-header",
)

layout = html.Div(
    [
        description_header,
        # dbc.Label(id="longcallback"),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3("Optimeringsindstillinger"),
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dbc.Form(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Ekstra årligt beløb",
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Input(
                                                                    type="number",
                                                                    id="ekstra_omkostning",
                                                                ),
                                                                dbc.FormText(
                                                                    id="eksisterende_omkostninger"
                                                                ),
                                                            ],
                                                            width=8,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Besparelse i CO2e",
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dbc.Input(
                                                                    type="number",
                                                                    id="reduktion_co2e",
                                                                ),
                                                                dbc.FormText(
                                                                    "Den ønskede procentvise CO2e besparelse"
                                                                ),
                                                            ],
                                                            width=8,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Prioritér CO2e besparelser",
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                dcc.Slider(
                                                                    min=0,
                                                                    max=10,
                                                                    step=1,
                                                                    value=5,
                                                                    marks=None,
                                                                    tooltip={
                                                                        "placement": "bottom",
                                                                        "always_visible": True,
                                                                    },
                                                                    id="co2e_prioritization",
                                                                ),
                                                                dbc.FormText(
                                                                    "Antal kroner du er villig til at betale for at spare X ton CO2e"
                                                                ),
                                                            ],
                                                            width=8,
                                                        ),
                                                    ],
                                                    className="mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Label(
                                                            "Vælg leasingperiode",
                                                            width=4,
                                                        ),
                                                        dbc.Col(
                                                            [
                                                                html.Div(
                                                                    id="leasingperiode"
                                                                ),
                                                                dbc.FormText(
                                                                    id="leasing_text"
                                                                ),
                                                            ],
                                                            width=8,
                                                        ),
                                                    ],
                                                    className="mb-4",
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                daq.BooleanSwitch(
                                                    on=False,
                                                    label={
                                                        "label": "Intelligent Allokering",
                                                        "style": {"font-size": "16px"},
                                                    },
                                                    color="#32CD32",
                                                    labelPosition="top",
                                                    id="intelligent_simulation_op",
                                                ),
                                                daq.BooleanSwitch(
                                                    on=False,
                                                    label={
                                                        "label": "Begræns km/år",
                                                        "style": {"font-size": "16px"},
                                                    },
                                                    color="#32CD32",
                                                    labelPosition="top",
                                                    id="km_aar_op",
                                                ),
                                            ],
                                            className="left-column-elements",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Optimér",
                                                    color="success",
                                                    className="right-btn",
                                                    id="optimer",
                                                    # disabled=True,
                                                )
                                            ],
                                            className="right-column-elements",
                                        ),
                                        html.Div([], id="vehicle-list"),
                                    ],
                                )
                            ]
                        ),
                    ]
                ),
                dbc.Col(
                    [
                        html.H3("Fremtidig flådesammensætning"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    html.H4("Total omkostning pr. år"),
                                                    html.P(id="solution_omkostning"),
                                                ],
                                                className="metric-card",
                                            )
                                        ]
                                    )
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    html.H4("CO2e-udledning pr. år"),
                                                    html.P(id="solution_udledning"),
                                                ],
                                                className="metric-card",
                                            )
                                        ]
                                    )
                                ),
                            ],
                            style={"margin-bottom": "10px"},
                        ),
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    card(
                                                        dcc.Graph(
                                                            figure=fig,
                                                            id={
                                                                "type": "goal_fig",
                                                                "index": 1,
                                                            },
                                                        ),
                                                    ),
                                                ),
                                                dbc.Col(
                                                    dbc.Card(
                                                        [
                                                            dbc.CardBody(
                                                                dcc.Graph(
                                                                    figure=fig,
                                                                    id={
                                                                        "type": "goal_fig",
                                                                        "index": 2,
                                                                    },
                                                                )
                                                            )
                                                        ]
                                                    ),
                                                ),
                                            ],
                                        ),
                                        html.Div(id="accordinan_solution"),
                                        dcc.Store(id="long_callback_optim"),
                                        dbc.Spinner(
                                            fullscreen=True,
                                            spinner_class_name="large_spinner",
                                            fullscreen_class_name="fullscreen_spinner",
                                            fullscreen_style={"visibility": "hidden"},
                                            id="optim_spinner",
                                            type="grow",
                                        ),
                                        dbc.Button("Afbryd Simulering", id="cancel_sim_goal", color="danger")
                                    ]
                                ),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    ],
)


@callback(Output("intelligent_simulation_op", "disabled"), Input("km_aar_op", "on"))
def enable_int_op(on):
    if on:
        return True
    return False


@callback(Output("km_aar_op", "disabled"), Input("intelligent_simulation_op", "on"))
def enable_km_op(on):
    if on:
        return True
    return False


@callback(
    Output({"type": "goal_fig_store", "index": MATCH}, "data"),
    Input({"type": "goal_fig", "index": MATCH}, "figure"),
)
def persistent_goal_figure(fig):
    return fig


@callback(
    Output("long_callback_optim", "data"),
    Input("optimer", "n_clicks"),
    Input("vehicle_sel_store", "data"),
    State("vehicle-list", "data"),
    State("location_store", "data"),
    State("date_store", "data"),
    State({"type": "goal_fig_store", "index": ALL}, "data"),
    State("solution_omkostning", "children"),
    State("solution_udledning", "children"),
    State("ekstra_omkostning", "value"),
    State("reduktion_co2e", "value"),
    State("co2e_prioritization", "value"),
    State("intelligent_simulation_op", "on"),
    State("km_aar_op", "on"),
)
def long_callback_optim(*args):
    if args[0] is None:
        return None
    return args


@app.long_callback(
    Output({"type": "goal_fig", "index": 1}, "figure"),
    Output({"type": "goal_fig", "index": 2}, "figure"),
    Output("accordinan_solution", "children"),
    Output("solution_udledning", "children"),
    Output("solution_udledning", "style"),
    Output("solution_omkostning", "children"),
    Output("solution_omkostning", "style"),
    Input("long_callback_optim", "data"),
    running=[
        (
            Output("optim_spinner", "fullscreen_style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        ),
        (
                Output("cancel_sim_goal", "style"),
                {"visibility": "visible"},
                {"visibility": "hidden"},
        )
    ],
    cancel=[Input("cancel_sim_goal", "n_clicks")],
    interval=3000,
    prevent_initial_call=True,
)
def optim(args):
    if args is None:
        return [fig, fig, [], 0, None, 0, None]
    (
        n_clicks,
        selected_vehicles,
        frame,
        location_id,
        dates,
        figs,
        omkostning,
        udledning,
        ekstra_omkostning,
        reduktion_co2e,
        co2e_prioritization,
        intelligent_simulation,
        km_aar,
    ) = args
    # todo persist figures
    # todo persist accordions
    # todo persist values
    if n_clicks is None or frame is None:
        return [*figs, [], 0, {}, 0, {}]
    active_vehicles = {}
    for vehicle in frame:
        if vehicle["id"] not in active_vehicles:
            active_vehicles[vehicle["id"]] = 0
        active_vehicles[vehicle["id"]] += vehicle["count"]
    settings = {
        "location": location_id,
        "dates": dates,
        "active_vehicles": active_vehicles,
    }
    fo = FleetOptimisation(settings, km_aar=km_aar)
    try:
        tb = TabuSearch(
            fo,
            location_id,
            dates,
            co2e_goal=0 if pd.isna(reduktion_co2e) else reduktion_co2e,
            expense_goal=0 if pd.isna(ekstra_omkostning) else ekstra_omkostning,
            weight=co2e_prioritization,
            intelligent=intelligent_simulation,
            km_aar=km_aar,
        )
    except RuntimeError:
        return [*figs, [], 0, {}, 0, {}]
    if len(tb.dummy_trips.trips) == 0:
        tb.report = []
    else:
        tb.run()
    if len(tb.report) == 0:
        return [*figs, [], 0, {}, 0, {}]

    omkostning = []
    co2e = []
    co2e_reduktioner = []
    omkostning_besparelse = []
    accordians = []
    keys = []
    header = html.Thead(
        html.Tr(
            [
                html.Th("Biltype"),
                html.Th("Drivmiddel forbrug"),
                html.Th("Årlig omk. (DKK)"),
                html.Th("Antal"),
            ]
        )
    )

    # walk through the solutions
    for k, solution in enumerate(tb.report):
        keys.append(f"Løsning {k + 1}")

        # save the solution details and generate accordian title
        omk = solution["omkostning"]
        omkostning.append(omk)
        udl = solution["co2e"]
        co2e.append(udl)
        title = f"Løsning {k + 1}: Årlig omkostning: {str(round(omk)).replace('.', ',')} DKK, Årlig ton CO2e: {str(round(udl, 3)).replace('.', ',')}"
        if solution["omkostning"] > tb.expense_goal or solution["co2e"] > tb.co2e_goal:
            if (
                solution["omkostning"] > tb.expense_goal
                and solution["co2e"] > tb.co2e_goal
            ):
                title += ". Overskredet omkostnings - og CO2e mål."
            else:
                title += (
                    ". Overskedet omkostningsmål."
                    if solution["omkostning"] > tb.expense_goal
                    else ". Overskredet CO2e mål."
                )

        fleet = []

        unique = {}
        for sv in solution['flåde']:
            name_emission = "__split__".join([sv['class_name'], sv['stringified_emission']])
            if name_emission not in unique:
                unique[name_emission] = {}
            car_omk = sv['omkostning_aar']
            if car_omk not in unique[name_emission]:
                unique[name_emission][car_omk] = 0
            unique[name_emission][car_omk] += sv['count']
        print(unique)
        # generate accordian vehicle table
        # for solution_vehicle in solution["flåde"]:
        #     fleet.append(
        #         html.Tr(
        #             [
        #                 html.Td(f'{solution_vehicle["class_name"]}'),
        #                 html.Td(solution_vehicle["omkostning_aar"]),
        #                 html.Td(solution_vehicle["stringified_emission"]),
        #                 html.Td(solution_vehicle["count"]),
        #             ]
        #         )
        #     )
        for car_type, omkostninger in unique.items():
            print(car_type)
            car_name, car_udledning = car_type.split("__split__")
            if len(omkostninger) == 1:
                fleet.append(
                    html.Tr(
                        [
                            html.Td(car_name),
                            html.Td(car_udledning),
                            html.Td(list(omkostninger.keys())[0]),
                            html.Td(list(omkostninger.values())[0])
                        ]
                    )
                )
            else:
                maj = []
                tot = 0
                for unique_omkostning, type_count in omkostninger.items():
                    maj.append(
                        html.Div(
                            [
                                # html.Td(car_name),
                                # html.Td(car_udledning),
                                html.Div("", className="col"),
                                html.Div("", className="col"),
                                html.Div("", className="col"),
                                html.Div("", className="col"),
                                html.Div(unique_omkostning, id="car_omkostning", className="col-md-auto"),
                                html.Div(type_count, id="unique_car_count", className="col")
                            ],
                        className="row")
                    )
                    tot += type_count
                ac = html.Tr(
                    html.Td(
                        dmc.Accordion(
                            dmc.AccordionItem(
                                maj,
                                label=[
                                    html.Div(
                                        [
                                            html.Div(car_name, className="col"),
                                            html.Div(car_udledning, className="col", id="car_udledning"),
                                            html.Div(html.Span(tot, id="car_count"), className="col")
                                        ],
                                        className="row")
                                ],
                                      ),
                            iconPosition="right",
                            class_name="accJohn",
                        ),
                        colSpan=4,
                    )
                )
                fleet.append(ac)



        # html.Div(
        #     children=[
        #         dmc.Accordion(
        #             id="accordion",
        #             state={"0": False, "1": True},
        #             children=[
        #                 dmc.AccordionItem(
        #                     "Content 1",
        #                     label="Section 1",
        #                 ),
        #                 dmc.AccordionItem(
        #                     "Content 2",
        #                     label="Section 2",
        #                 ),
        #             ],
        #         ),
        #         dmc.Text(id="accordion-state", style={"marginTop": 10}),
        #     ]
        # )










        # details for the "Simuleringsdetaljer" table
        besparelse = round(tb.cur_result[0]) - round(solution["omkostning"])
        omkostning_besparelse.append(besparelse)
        reduktion = round(tb.cur_result[1] - solution["co2e"], 3)
        co2e_reduktioner.append(reduktion)
        reduktion = f"{str(reduktion).replace('.', ',')}"
        solution_table = accordian_table(
            header,
            fleet,
            tb.cur_result[0],
            solution["omkostning"],
            besparelse,
            tb.cur_result[1],
            solution["co2e"],
            reduktion,
        )

        accordians.append(dbc.AccordionItem(solution_table, title=title))

    accordian_card = dbc.Accordion(accordians, always_open=True)

    omkostning_bar = goal_bar_plot(
        keys,
        omkostning,
        title="Årlig omkostning i kr.",
        xlabel="Løsninger",
        ylabel="kr. pr. år",
    )
    co2e_bar = goal_bar_plot(
        keys,
        co2e,
        title="Årlig udledning ton CO2e",
        xlabel="Løsninger",
        ylabel="Ton CO2e pr. år",
    )

    return [
        omkostning_bar,
        co2e_bar,
        accordian_card,
        f"{str(round(co2e[0], 3)).replace('.', ',')} Ton CO2e",
        {"color": "red" if co2e_reduktioner[0] < 0 else "green"},
        f"{round(omkostning[0])} DKK",
        {"color": "red" if omkostning_besparelse[0] < 0 else "green"},
    ]


@callback(
    Output("leasingperiode", "children"),
    Output("eksisterende_omkostninger", "children"),
    State("date_store", "data"),
    State("location_store", "data"),
    Input("url", "pathname"),
    Input("vehicle_sel_store", "data"),
)
def make_vehicle_list(dates, location, url, selected_vehicles):
    # todo ensure to pull the bikes
    if url != "/page_goal" or selected_vehicles is None:
        raise PreventUpdate
    omkostninger = "Det ekstra årlige beløb du vil betale."
    start = datetime.datetime.now()
    current_expense = 0
    with Session() as session:
        expense_query = session.query(
            func.sum(Cars.omkostning_aar).label("total_expense")
        ).filter(
            Cars.location == location,
            func.coalesce(Cars.end_leasing, dates[-1]) >= dates[-1],
        )
        for a in expense_query:
            current_expense += a.total_expense

        omkostninger += f" Nuværende faste omkostninger: {round(current_expense)} DKK, eksl. brændstofforbrug"
        rows = session.query(
            func.coalesce(Cars.end_leasing, start).label("end_leasing")
        ).filter(Cars.id.in_(selected_vehicles))
        dates = []
        for row in rows:
            dates.append(row.end_leasing)
    if dates is None or len(dates) == 0:
        all_dates = [
            datetime.datetime.now(),
            datetime.datetime.now() + datetime.timedelta(days=365),
        ]
    else:
        sorted_dates = sorted(dates)
        all_dates = (
            [datetime.datetime.now()]
            + [
                datetime.datetime.fromisoformat(date) if type(date) is str else date
                for date in sorted(sorted_dates)
            ]
            + [
                datetime.datetime.fromisoformat(sorted_dates[-1])
                if type(sorted_dates[-1]) is str
                else sorted_dates[-1]
            ]
        )
        all_dates[-1] += datetime.timedelta(days=30)
    value = [unix_time_millis(date) for date in all_dates]
    marks = {
        a: str(date.date())
        for a, date in zip([value[0], value[-1]], [all_dates[0], all_dates[-1]])
    }
    mind = value[0]
    maxd = value[-1]

    value = [value[0], value[-1]]
    date_slider = (
        dcc.RangeSlider(
            min=mind,
            max=maxd,
            value=value,
            marks=marks,
            tooltip={
                "always_visible": False,
            },
            id="slids",
        ),
    )
    return [date_slider, omkostninger]


@callback(
    Output("vehicle-list", "children"),
    Output("vehicle-list", "data"),
    Output("leasing_text", "children"),
    Input("slids", "value"),
    Input("vehicle_sel_store", "data"),
)
def update_view(unix_times, selected_vehicles):
    start, end = [unix_to_datetime(va) for va in unix_times]
    leasing_text = f"Juster leasingperioden hvis du ønsker at frigøre køretøjer til udskiftning. Valgt periode: {start.date()} til {end.date()}"
    with Session() as session:
        fake_end = datetime.datetime(year=1000, month=1, day=1)
        subq = (
            session.query(
                Cars.make,
                func.coalesce(Cars.model, "").label("model"),
                func.coalesce(Cars.omkostning_aar, 0).label("omkostning_aar"),
                func.coalesce(Cars.co2_pr_km, 0).label("co2_pr_km"),
                func.coalesce(Cars.km_aar, 0).label("km_aar"),
                func.coalesce(Cars.sleep, 0).label("sleep"),
                func.coalesce(Cars.range, 0).label("range"),
                func.coalesce(Cars.wltp_el, float(0)).label("wltp_el"),
                func.coalesce(Cars.wltp_fossil, float(0)).label("wltp_fossil"),
                func.coalesce(Cars.capacity_decrease, float(0)).label(
                    "capacity_decrease"
                ),
                func.count(Cars.make).label("count"),
                func.coalesce(Cars.end_leasing, fake_end).label("end_leasing"),
            )
            .filter(
                Cars.id.in_(selected_vehicles),
                or_(
                    Cars.end_leasing == fake_end,
                    and_(
                        func.coalesce(Cars.end_leasing, end) <= end,
                        func.coalesce(Cars.end_leasing, end) >= start,
                    ),
                ),
            )
            .group_by(
                Cars.make,
                Cars.model,
                Cars.omkostning_aar,
                Cars.co2_pr_km,
                Cars.km_aar,
                Cars.sleep,
                Cars.range,
                Cars.wltp_el,
                Cars.wltp_fossil,
                Cars.capacity_decrease,
                Cars.end_leasing,
            )
        ).subquery()
        cte = (
            session.query(Cars.id, subq)
            .filter(Cars.id.in_(selected_vehicles))
            .join(
                Cars,
                (Cars.make == subq.c.make)
                & (func.coalesce(Cars.model, "") == subq.c.model)
                & (func.coalesce(Cars.omkostning_aar, 0) == subq.c.omkostning_aar)
                & (func.coalesce(Cars.end_leasing, fake_end) == subq.c.end_leasing)
                & (func.coalesce(Cars.co2_pr_km, 0) == subq.c.co2_pr_km)
                & (func.coalesce(Cars.km_aar, 0) == subq.c.km_aar)
                & (func.coalesce(Cars.sleep, 0) == subq.c.sleep)
                & (func.coalesce(Cars.range, 0) == subq.c.range)
                & (func.coalesce(Cars.wltp_el, float(0)) == subq.c.wltp_el)
                & (func.coalesce(Cars.wltp_fossil, float(0)) == subq.c.wltp_fossil)
                & (
                    func.coalesce(Cars.capacity_decrease, float(0))
                    == subq.c.capacity_decrease
                ),
            )
            .cte()
        )
        first_id = (
            session.query(func.min(cte.c.id))
            .filter(cte.c.id.in_(selected_vehicles))
            .group_by(
                cte.c.make,
                cte.c.model,
                cte.c.omkostning_aar,
                cte.c.co2_pr_km,
                cte.c.km_aar,
                cte.c.sleep,
                cte.c.range,
                cte.c.wltp_el,
                cte.c.wltp_fossil,
                cte.c.capacity_decrease,
                cte.c.end_leasing,
            )
        )
        rows = session.query(cte).filter(cte.c.id.in_(first_id))
        vehicle_data = []
        body = []
        for row in rows:
            vehicle_data.append(
                {
                    "id": row.id,
                    "make": row.make,
                    "model": row.model,
                    "omkostning_aar": row.omkostning_aar,
                    "co2_pr_km": row.co2_pr_km,
                    "km_aar": row.km_aar,
                    "sleep": row.sleep,
                    "range": row.range,
                    "wltp_el": row.wltp_el,
                    "wltp_fossil": row.wltp_fossil,
                    "capacity_decrease": row.capacity_decrease,
                    "count": row.count,
                    "end_leasing": row.end_leasing,
                }
            )
            body.append(
                html.Tr(
                    [
                        html.Td("{} {}".format(row.make, row.model)),
                        html.Td(round(row.omkostning_aar)),
                        html.Td(get_emission(row)),
                        html.Td(row.count),
                        html.Td(
                            "Ejet"
                            if str(row.end_leasing) == str(fake_end)
                            else str(row.end_leasing).split()[0]
                        ),
                    ]
                )
            )

        header = html.Thead(
            html.Tr(
                [
                    html.Th("Biltype"),
                    html.Th("Årlig omk. (DKK)"),
                    html.Th("Drivmiddel forbrug"),
                    html.Th("Antal i beholdning"),
                    html.Th("Slut leasing"),
                ]
            )
        )

        return [
            dbc.Table(
                [header, html.Tbody(body)],
            ),
            vehicle_data,
            leasing_text,
        ]
