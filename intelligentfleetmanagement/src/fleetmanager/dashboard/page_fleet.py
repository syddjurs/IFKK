import datetime
import io
import json

import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, MATCH, Input, Output, State, callback, callback_context, dcc, html
from dash.exceptions import PreventUpdate
from sqlalchemy import func

import fleetmanager.data_access.db_engine as db
from ..data_access import Cars
from ..model import Model
from .Components.time_picker import MultiTimePickerAIO
from .utils import (
    capacity_plot_from,
    card,
    distance_histogram_from,
    get_emission,
    input_table,
    prepare_trip_store,
)
from .view import app

Session = db.session_factory(db.engine_creator())
fig = go.Figure()

modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Tilføj Køretøj")),
        dbc.ModalBody(id="fleet_add"),
    ],
    id="fleet_modal",
)

description_header = dbc.Toast(
    [
        html.P(
            "På denne side kan man som bruger undersøge hvilken indflydelse det har, at tilføje eller fjerne køretøjer "
            "til den samlede flåde. Disse ændringer kommer til at påvirke de estimerede årlige omkostninger, "
            "det estimerede årlige CO2e forbrug, samt hvor mange ture der ikke bliver allokeret.",
            className="mb-0",
        )
    ],
    header="Flådesammensætning",
    className="description-header",
)

layout = html.Div(
    [
        description_header,
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H3(id="antal_descriptor"),
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            id="fleet_view",
                                            className="fleet-view",
                                        ),
                                        html.Hr(),
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
                                                    id="intelligent_simulation",
                                                ),
                                                daq.BooleanSwitch(
                                                    on=False,
                                                    label={
                                                        "label": "Begræns km/år",
                                                        "style": {"font-size": "16px"},
                                                    },
                                                    color="#32CD32",
                                                    labelPosition="top",
                                                    id="km_aar",
                                                ),
                                            ],
                                            className="left-column-elements",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Button(
                                                    "Konfigurér cykler",
                                                    color="secondary",
                                                    id="config-bike",
                                                ),
                                                dbc.Button(
                                                    "Tilføj køretøj",
                                                    color="primary",
                                                    id="fleet_add_btn",
                                                ),
                                            ],
                                            className="right-column-elements",
                                        ),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader(
                                                    dbc.ModalTitle(
                                                        "Konfigurér cykel egenskaber"
                                                    )
                                                ),
                                                dbc.ModalBody(
                                                    [
                                                        dbc.Form(
                                                            [
                                                                dbc.Label(
                                                                    "Maks. km. pr. tur"
                                                                ),
                                                                dbc.Input(
                                                                    type="number",
                                                                    id="bike-max-km-per-trip",
                                                                    value=5,
                                                                ),
                                                                dbc.Label(
                                                                    "Procent af disse ture som køres"
                                                                ),
                                                                dbc.Input(
                                                                    type="number",
                                                                    min=0,
                                                                    max=100,
                                                                    id="bike-percent-of-trips",
                                                                ),
                                                                MultiTimePickerAIO(
                                                                    "Tidsrum hvor cykler kan tildeles"
                                                                ),
                                                                dbc.Button(
                                                                    "Ok",
                                                                    id="config-bike-ok",
                                                                    color="primary",
                                                                    className="right-btn",
                                                                ),
                                                            ]
                                                        )
                                                    ]
                                                ),
                                            ],
                                            id="bike-config-modal",
                                            size="lg",
                                            is_open=False,
                                        ),
                                    ]
                                )
                            ]
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        html.H3("Overblik over simuleringsresultater"),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "Rundture uden køretøj",
                                                            id="sim_daterange",
                                                        ),
                                                        html.P(
                                                            id="trips_without_vehicle",
                                                            style=None,
                                                        ),
                                                    ],
                                                    className="metric-card",
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4("Besparelse pr. år"),
                                                        html.P(
                                                            id="savings_pr_year",
                                                            style=None,
                                                        ),
                                                    ],
                                                    className="metric-card",
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Card(
                                            [
                                                dbc.CardBody(
                                                    [
                                                        html.H4(
                                                            "Reduktion i udledning pr. år"
                                                        ),
                                                        html.P(
                                                            id="savings_pr_year_co2e",
                                                            style=None,
                                                        ),
                                                    ],
                                                    className="metric-card",
                                                )
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="bot-margin",
                        ),
                        dbc.Row(
                            dbc.Col(
                                [
                                    html.Div(
                                        [
                                            html.A(
                                                [
                                                    html.I(
                                                        className="fas fa-download",
                                                        style={
                                                            "vertical-align": "middle"
                                                        },
                                                    ),
                                                    html.Span(
                                                        "Download resultater til Excel",
                                                        style={"margin-left": "5px"},
                                                    ),
                                                ],
                                                id="download_excel_button",
                                                download="True",
                                                className="download-link",
                                            ),
                                        ],
                                        className="flex-container",
                                    ),
                                    dcc.Download(id="download-results"),
                                ]
                            )
                        ),
                        dbc.Row(
                            dbc.Col(
                                card(
                                    dcc.Graph(
                                        figure=fig, id={"type": "fig", "index": 1}
                                    ),
                                )
                            ),
                        ),
                        html.Br(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                dcc.Graph(
                                                    figure=fig,
                                                    id={"type": "fig", "index": 2},
                                                )
                                            )
                                        ]
                                    ),
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                dcc.Graph(
                                                    figure=fig,
                                                    id={"type": "fig", "index": 3},
                                                )
                                            )
                                        ]
                                    ),
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        html.Br(),
        html.Div(
            [
                # dbc.Button(
                #     "Intelligent turallokering",
                #     color="secondary",
                # ),
                dbc.Button(
                    "Simuler",
                    id="btn_sim",
                    color="success",
                ),
            ],
            className="flex-container",
        ),
        modal,
        dcc.Store(id="lcb_in"),
        dcc.Store(id="sim_count_store"),
        dbc.Spinner(
            id="sim_spinner",
            fullscreen=True,
            spinner_class_name="large_spinner",
            fullscreen_class_name="fullscreen_spinner",
            fullscreen_style={"visibility": "hidden"},
            type="grow",
        ),
    ]
)


@callback(
    Output("download-results-csv", "data"),
    Input("download_csv_button", "n_clicks"),
    State("trip_store", "data"),
    State("location_name", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, trip_store, location_name):
    if trip_store == "":
        return None
    return dcc.send_data_frame(
        pd.DataFrame(trip_store).to_csv,
        filename=f"results_{location_name[0].split(',')[0]}.csv",
        index=False,
    )


@callback(
    Output("download-results", "data"),
    Input("download_excel_button", "n_clicks"),
    State({"type": "fig", "index": ALL}, "figure"),
    State("trip_store", "data"),
    State("location_name", "data"),
    prevent_initial_call=True,
)
def download(n_clicks, figures, trip_store, location_name):
    # Don't download if figures have no data
    if not figures[0]["data"]:
        return None

    un_allocated_trips_data = figures[0]["data"]
    timestamps = un_allocated_trips_data[0]["x"]
    current = un_allocated_trips_data[0]["y"]
    simulated = un_allocated_trips_data[1]["y"]
    temp_frame = [
        [timestamps[i], current[i], simulated[i]] for i in range(len(timestamps))
    ]
    un_allocated_trips_data = pd.DataFrame(
        temp_frame, columns=["Timestamp", "Current", "Simulated"]
    )

    def extract_df_from_bar_graph(graph):
        bucket_size_half = graph[0]["x"][0]
        distances = [
            round(distance - bucket_size_half, 1) for distance in graph[0]["x"]
        ]
        distances.append(round(graph[0]["x"][-1] + bucket_size_half, 1))
        headers = ["Køretøjstyper"] + [
            f"{distances[i]} km - {distances[i+1]} km"
            for i in range(len(distances) - 1)
        ]
        headers.append(f"{round(distances[-1]+bucket_size_half, 1)}+ km")
        temp_frame = []
        for row in graph:
            temp_frame.append([row["name"]] + row["y"])
        return pd.DataFrame(temp_frame, columns=headers)

    allocated_current = extract_df_from_bar_graph(figures[1]["data"])
    allocated_simulated = extract_df_from_bar_graph(figures[2]["data"])
    driving_plan = pd.DataFrame(trip_store)
    output = io.BytesIO()
    with pd.ExcelWriter(output) as writer:
        driving_plan.to_excel(writer, sheet_name="Køreplan", index=False)
        un_allocated_trips_data.to_excel(
            writer, sheet_name="Ikke allokerede ture", index=False
        )
        allocated_current.to_excel(
            writer, sheet_name="Nuværende allokering", index=False
        )
        allocated_simulated.to_excel(
            writer, sheet_name="Simuleret allokering", index=False
        )
        writer.save()
        data = output.getvalue()
    return dcc.send_bytes(
        data,
        filename=f"results_{location_name[0].split(',')[0]}.xlsx",
    )


@callback(
    Output("bike-config-modal", "is_open"),
    Input("config-bike", "n_clicks"),
    Input("config-bike-ok", "n_clicks"),
    [State("bike-config-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


@callback(
    Output("fleet_modal", "is_open"),
    Input("fleet_add_btn", "n_clicks"),
    prevent_initial_call=True,
)
def open_modal(n):
    return True


@callback(
    Output("fleet_view", "children"),
    Output("vehicle_idx_store", "data"),
    Input("url", "pathname"),
    Input({"type": "fleet_values", "index": ALL}, "value"),
    Input({"type": "fleet_values", "index": ALL}, "name"),
    State("vehicle_sel_store", "data"),
    prevent_initial_call=True,
)
def fleet_view(url, checklist, id_checklist, selected_vehicles):
    # todo initial callback prompts error in debug mode, but runs alright
    if url != "/page_fleet" or selected_vehicles is None or checklist is None:
        raise PreventUpdate

    all_vehicles = []
    style = []

    extra_vehicles = [id for id, check in zip(id_checklist, checklist) if check]

    with Session() as session:
        subq = (
            session.query(
                Cars.make,
                func.coalesce(Cars.model, "").label("model"),
                Cars.omkostning_aar,
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
            )
            .filter((Cars.id.in_(selected_vehicles)))
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
            )
            .subquery()
        )
        cte = (
            session.query(Cars.id, subq)
            .filter(Cars.id.in_(selected_vehicles))
            .join(
                Cars,
                (Cars.make == subq.c.make)
                & (func.coalesce(Cars.model, "") == subq.c.model)
                & (Cars.omkostning_aar == subq.c.omkostning_aar)
                & (func.coalesce(Cars.co2_pr_km, 0) == subq.c.co2_pr_km)
                & (func.coalesce(Cars.km_aar, 0) == subq.c.km_aar)
                & (func.coalesce(Cars.sleep, 0) == subq.c.sleep)
                & (func.coalesce(Cars.range, 0) == subq.c.range)
                & (func.coalesce(Cars.wltp_el, float(0)) == subq.c.wltp_el)
                & (func.coalesce(Cars.wltp_fossil, float(0)) == subq.c.wltp_fossil)
                & (
                    func.coalesce(Cars.capacity_decrease, 0) == subq.c.capacity_decrease
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
            )
        )
        current_selected_cars = session.query(cte).filter(cte.c.id.in_(first_id))

        for vehicle in current_selected_cars:
            all_vehicles.append(
                [
                    vehicle.id,
                    "{} {}".format(vehicle.make, vehicle.model),
                    vehicle.omkostning_aar,
                    get_emission(vehicle),
                    vehicle.count,
                ]
            )
            style.append({})

        extra_vehicles = session.query(
            Cars.id,
            Cars.make,
            Cars.model,
            Cars.omkostning_aar,
            Cars.co2_pr_km,
            Cars.wltp_el,
            Cars.wltp_fossil,
        ).filter(Cars.id.in_(extra_vehicles))

        for vehicle in extra_vehicles:
            all_vehicles.append(
                [
                    vehicle.id,
                    "{} {}".format(vehicle.make, vehicle.model),
                    vehicle.omkostning_aar,
                    get_emission(vehicle),
                    0,
                ]
            )
            style.append({"color": "#109CF1"})
    return (
        html.Div(
            input_table(
                all_vehicles,
                header=[
                    "Biltype",
                    "Årlige Omkostninger (DKK)",
                    "Drivmiddel forbrug",
                    "Nuværende antal",
                    "Antal i simulation",
                ],
                id="vehicle_values",
                col_style=style,
                fn_input=dbc.Input,
                fn_params={
                    "value": 0,
                    "min": 0,
                    "type": "number",
                },
                total_id="n_total_sim",
            ),
            className="styled-numeric-input",
            id="fleet_view",
        ),
        json.dumps(all_vehicles),
    )


@callback(
    Output({"type": "fig_store", "index": MATCH}, "data"),
    Input({"type": "fig", "index": MATCH}, "figure"),
)
def persistent_figures(fig):
    return fig


@callback(Output("intelligent_simulation", "disabled"), Input("km_aar", "on"))
def enable_int(on):
    if on:
        return True
    return False


@callback(Output("km_aar", "disabled"), Input("intelligent_simulation", "on"))
def enable_km(on):
    if on:
        return True
    return False


@callback(
    Output("sim_daterange_store", "data"),
    Input("sim_daterange", "children"),
)
def persistent_daterange(children):
    return children


@callback(
    Output("trips_without_vehicle_store", "data"),
    Input("trips_without_vehicle", "children"),
    State("trips_without_vehicle", "style"),
)
def persistent_trips_without_vehicle(children, style):
    return children, style


@callback(
    Output("savings_pr_year_store", "data"),
    Input("savings_pr_year", "children"),
    State("savings_pr_year", "style"),
)
def persistent_savings_pr_year(children, style):
    return children, style


@callback(
    Output("savings_pr_year_co2e_store", "data"),
    Input("savings_pr_year_co2e", "children"),
    State("savings_pr_year_co2e", "style"),
)
def persistent_savings_pr_year_co2e(children, style):
    return children, style


@callback(
    Output("n_total_sim", "children"),
    Input({"type": "vehicle_values", "index": ALL}, "value"),
)
def update_total_sim(n_vehicle):
    try:
        return sum(n_vehicle)
    except TypeError:
        raise PreventUpdate


@callback(
    Output("lcb_in", "data"),
    Input("url", "pathname"),
    Input("btn_sim", "n_clicks"),
    State("intelligent_simulation", "on"),
    State("km_aar", "on"),
    State({"type": "vehicle_values", "index": ALL}, "value"),
    State("vehicle_sel_store", "data"),
    State("fleet_store", "data"),
    State({"type": "fleet_values", "index": ALL}, "value"),
    State({"type": "fig_store", "index": ALL}, "data"),
    State("date_store", "data"),
    State("location_store", "data"),
    State("bike-max-km-per-trip", "value"),
    State("bike-percent-of-trips", "value"),
    State(
        {"component": "MultiTimePickerAIO", "type": "bike_start", "id": ALL}, "value"
    ),
    State({"component": "MultiTimePickerAIO", "type": "bike_end", "id": ALL}, "value"),
    State("vehicle_idx_store", "data"),
    State("location_name", "data"),
    State("sim_daterange_store", "data"),
    State("trips_without_vehicle_store", "data"),
    State("savings_pr_year_store", "data"),
    State("savings_pr_year_co2e_store", "data"),
    State("trip_store", "data"),
)
def lcb_in(*args):
    if args[1] is None:
        return [None, args[-6]]
    ctx = callback_context
    return args + tuple([ctx.triggered[0]])


@app.long_callback(
    Output({"type": "fig", "index": 1}, "figure"),
    Output({"type": "fig", "index": 2}, "figure"),
    Output({"type": "fig", "index": 3}, "figure"),
    Output("sim_daterange", "children"),
    Output("trips_without_vehicle", "children"),
    Output("trips_without_vehicle", "style"),
    Output("savings_pr_year", "children"),
    Output("savings_pr_year", "style"),
    Output("savings_pr_year_co2e", "children"),
    Output("savings_pr_year_co2e", "style"),
    Output("antal_descriptor", "children"),
    Output("trip_store", "data"),
    Input("lcb_in", "data"),
    running=[
        (
            Output("sim_spinner", "fullscreen_style"),
            {"visibility": "visible"},
            {"visibility": "hidden"},
        )
    ],
    prevent_initial_call=True,
)
def simulate(args):
    if args[0] is None:
        return (
            fig, fig, fig,
            "Rundture uden køretøj",
            0, None,
            0, None,
            0, None,
            "Antal af køretøjer"
            if pd.isna(args[1])
            else f"Antal af køretøjer ved {args[1][0]}",
            "",
        )
    (
        pathname,
        n_clicks,
        intelligent_simulation,
        km_aar,
        n_vehicle,
        vehicle_sel_store,
        fleet_store,
        mask,
        figs,
        date_range,
        location_id,
        bike_max_distance,
        bike_percentage,
        bike_start_times,
        bike_end_times,
        vis,
        location_name,
        sim_daterange0,
        trips_without_vehicle0,
        savings_pr_year0,
        savings_pr_year_co2e0,
        trip_store0,
        triggered0,
    ) = args

    if None in (
        sim_daterange0,
        trips_without_vehicle0,
        savings_pr_year0,
        savings_pr_year_co2e0,
        trip_store0,
    ):
        sim_daterange0 = "Rundture uden køretøj"
        trips_without_vehicle0 = ["", None]
        savings_pr_year0 = ["", None]
        savings_pr_year_co2e0 = ["", None]
        trip_store0 = ""

    antal_description = (
        "Antal af køretøjer"
        if location_name is None
        else f"Antal af køretøjer ved {location_name[0]}"
    )

    if (
        (
            pathname == "/page_fleet"
            and triggered0["prop_id"] == "."
            or (date_range is None or location_id is None or vis is None)
        )
        or (pathname != "/page_fleet" and triggered0["prop_id"] != "btn_sim.n_clicks")
        or (pathname == "/page_fleet" and triggered0["value"] is None)
    ):
        return (
            *figs,
            sim_daterange0,
            *trips_without_vehicle0,
            *savings_pr_year0,
            *savings_pr_year_co2e0,
            antal_description,
            trip_store0,
        )

    m = Model(location=location_id, dates=date_range)

    bike_time_slots = [
        (
            datetime.datetime.strptime(start, "%H:%M"),
            datetime.datetime.strptime(end, "%H:%M"),
        )
        for start, end in zip(bike_start_times, bike_end_times)
    ]
    max_bike_time_slot = max(
        [(end - start).total_seconds() / 3600 for start, end in bike_time_slots] + [0]
    )
    bike_percentage = 100 if pd.isna(bike_percentage) else bike_percentage

    # Need to tell users that things haven't been updated
    # TODO: fix this plz
    if len(m.trips.trips) == 0:
        return (
            *figs,
            sim_daterange0,
            *trips_without_vehicle0,
            *savings_pr_year0,
            *savings_pr_year_co2e0,
            antal_description,
            trip_store0,
        )

    indices = m.fleet_manager.vehicle_factory.all_vehicles
    # temporary hack until we start using orm in the model
    # match id's of selected vehicles to indices and get index of selected vehicles
    aggregated_vehicles = json.loads(vis)
    selected_vehicles = indices[indices.id.isin(vehicle_sel_store)].index.values
    for index in selected_vehicles:
        setattr(m.fleet_manager.current_fleet, str(index), 1)

    counter = {}
    for selected_vehicle, n in zip(aggregated_vehicles, n_vehicle):
        ind = str(indices.loc[indices["id"] == selected_vehicle[0]].index[0])
        if ind not in counter:
            counter[ind] = 0
        counter[ind] += n
    for vehicle_index, vehicle_count in counter.items():
        setattr(m.fleet_manager.simulation_fleet, vehicle_index, vehicle_count)

    m.run_simulation(
        intelligent_simulation,
        bike_max_distance,
        bike_time_slots,
        max_bike_time_slot,
        bike_percentage,
        km_aar,
    )
    twv = sum(m.capacity_source.data["sim_unassigned_trips"])
    dato = "Rundture uden køretøj"
    if len(date_range) > 0:
        dato = (
            f"Rundture uden køretøj fra {datetime.datetime.fromisoformat(date_range[0]).date()} til "
            f"{datetime.datetime.fromisoformat(date_range[1]).date()}"
        )
    savings_key = m.consequence_source.data["keys"].index("Samlet omkostning [kr/år]")
    co2e_key = m.consequence_source.data["keys"].index(
        "POGI CO2-ækvivalent udledning [CO2e]"
    )
    savings = round(
        m.consequence_source.data["cur_values"][savings_key]
        - m.consequence_source.data["sim_values"][savings_key],
    )
    co2e_savings = round(
        m.consequence_source.data["cur_values"][co2e_key]
        - m.consequence_source.data["sim_values"][co2e_key],
        3,
    )
    trip_store = prepare_trip_store(m.trips.trips, location_name)

    return (
        capacity_plot_from(
            m.capacity_source,
            "Tidspunkt",
            "Antal ture uden køretøj",
            "Tidspunkter med ikke-fordelte ture",
        ),
        distance_histogram_from(
            m.current_hist,
            "Turlængde (km)",
            "Antal",
            "Turfordeling for nuværende flåde",
        ),
        distance_histogram_from(
            m.simulation_hist,
            "Turlængde (km)",
            "Antal",
            "Turforderling for simuleret flåde",
        ),
        dato,
        twv,
        {"color": "red" if twv else "green"},
        f"{str(savings).replace('.', ',')} DKK",
        {"color": "red" if savings < 0 else "green"},
        f"{str(co2e_savings).replace('.', ',')} ton CO2e",
        {"color": "red" if co2e_savings < 0 else "green"},
        antal_description,
        trip_store,
    )


@callback(
    Output("fleet_add", "children"),
    Output("fleet_store", "data"),
    Input("fleet_add_btn", "n_clicks"),
    State("vehicle_sel_store", "data"),
)
def fleet_add(n, selected_vehicles):
    if not selected_vehicles:
        raise PreventUpdate
    vehicle_info = []
    indices = []
    with Session() as session:
        subq = (
            session.query(
                Cars.make,
                func.coalesce(Cars.model, "").label("model"),
                Cars.omkostning_aar,
                func.coalesce(Cars.co2_pr_km, 0).label("co2_pr_km"),
                func.coalesce(Cars.km_aar, 0).label("km_aar"),
                func.coalesce(Cars.sleep, 0).label("sleep"),
                func.coalesce(Cars.range, 0).label("range"),
                func.coalesce(Cars.wltp_el, float(0)).label("wltp_el"),
                func.coalesce(Cars.wltp_fossil, float(0)).label("wltp_fossil"),
                func.coalesce(Cars.capacity_decrease, float(0)).label(
                    "capacity_decrease"
                ),
            )
            .filter(
                (Cars.id.notin_(selected_vehicles)) & (Cars.omkostning_aar.isnot(None))
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
            )
            .subquery()
        )
        cte = (
            session.query(Cars.id, subq)
            .filter(Cars.id.notin_(selected_vehicles))
            .join(
                Cars,
                (Cars.make == subq.c.make)
                & (func.coalesce(Cars.model, "") == subq.c.model)
                & (Cars.omkostning_aar == subq.c.omkostning_aar)
                & (func.coalesce(Cars.co2_pr_km, 0) == subq.c.co2_pr_km)
                & (func.coalesce(Cars.km_aar, 0) == subq.c.km_aar)
                & (func.coalesce(Cars.sleep, 0) == subq.c.sleep)
                & (func.coalesce(Cars.range, 0) == subq.c.range)
                & (func.coalesce(Cars.wltp_el, float(0)) == subq.c.wltp_el)
                & (func.coalesce(Cars.wltp_fossil, float(0)) == subq.c.wltp_fossil)
                & (
                    func.coalesce(Cars.capacity_decrease, 0) == subq.c.capacity_decrease
                ),
            )
            .cte()
        )
        first_id = (
            session.query(func.min(cte.c.id))
            .filter(cte.c.id.notin_(selected_vehicles))
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
            )
        )
        rows = session.query(cte).filter(cte.c.id.in_(first_id))
        current_vehicles = (
            session.query(
                Cars.make,
                Cars.model,
                Cars.omkostning_aar,
                func.coalesce(Cars.co2_pr_km, 0).label("co2_pr_km"),
                func.coalesce(Cars.km_aar, 0).label("km_aar"),
                func.coalesce(Cars.sleep, 0).label("sleep"),
                func.coalesce(Cars.range, 0).label("range"),
                func.coalesce(Cars.wltp_el, float(0)).label("wltp_el"),
                func.coalesce(Cars.wltp_fossil, float(0)).label("wltp_fossil"),
                func.coalesce(Cars.capacity_decrease, float(0)).label(
                    "capacity_decrease"
                ),
            )
            .filter(Cars.id.in_(selected_vehicles))
            .all()
        )
        for row in rows:
            if row[1:] in current_vehicles:
                continue
            vehicle_info.append(
                [
                    row.id,
                    "{} {}".format(row.make, row.model),
                    row.omkostning_aar,
                    get_emission(row),
                ]
            )
            indices.append(row.id)

    return (
        input_table(
            vehicle_info,
            header=[
                "Biltype",
                "Omkostninger (DKK)",
                "Drivmiddel forbrug",
            ],
            id="fleet_values",
            fn_input=dbc.Checkbox,
        ),
        indices,
    )
