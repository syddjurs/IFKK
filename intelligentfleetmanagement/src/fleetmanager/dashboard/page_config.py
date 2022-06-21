from datetime import date, datetime

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dash_table, dcc, html
from dash.dash_table.Format import Format, Group, Scheme, Symbol, Trim
from dash.exceptions import PreventUpdate
from sqlalchemy.orm.query import Query

from fleetmanager import data_access
from fleetmanager.data_access import (
    AllowedStarts,
    Cars,
    FuelTypes,
    LeasingTypes,
    VehicleTypes,
    session_factory,
)

engine = data_access.db_engine.engine_creator()
Session = session_factory(engine)


def get_cars_rounded():
    data = pd.read_sql(Query([data_access.dbschema.Cars]).statement, engine)
    data["start_leasing"] = pd.to_datetime(data["start_leasing"]).dt.date
    data["end_leasing"] = pd.to_datetime(data["end_leasing"]).dt.date
    return data.to_dict("records")


def init_table():

    with Session() as s:
        types = s.query(VehicleTypes).all()
        fuel = s.query(FuelTypes).all()
        leasingtypes = s.query(LeasingTypes).all()
        locations = s.query(AllowedStarts).all()

    configuration_table = dash_table.DataTable(
        id="config_table",
        style_as_list_view=True,
        data=get_cars_rounded(),
        columns=[
            {"id": "id", "name": "Id", "type": "numeric", "editable": False},
            {"id": "plate", "name": "Nummerplade", "type": "text"},
            {"id": "make", "name": "Mærke", "type": "text"},
            {"id": "model", "name": "Model", "type": "text"},
            {"id": "type", "name": "Type", "presentation": "dropdown"},
            {"id": "fuel", "name": "Drivmiddel", "presentation": "dropdown"},
            {"id": "wltp_fossil", "name": "Wltp (fossil)", "type": "numeric",},
            {"id": "wltp_el", "name": "Wltp (el)", "type": "numeric",},
            {"id": "capacity_decrease", "name": "Procentvis wltp", "type": "numeric",},
            {"id": "co2_pr_km", "name": "CO2 (g/km)", "type": "numeric",},
            {"id": "range", "name": "Rækkevidde (km)", "type": "numeric",},
            {"id": "omkostning_aar", "name": "Omk./år", "type": "numeric",},
            {"id": "location", "name": "Lokation", "presentation": "dropdown"},
            {"id": "start_leasing", "name": "Start leasing", "type": "datetime"},
            {"id": "end_leasing", "name": "Slut leasing", "type": "datetime"},
            {"id": "leasing_type", "name": "Leasingtype", "presentation": "dropdown"},
            {"id": "km_aar", "name": "Tilladt km/år", "type": "numeric",},
            {"id": "sleep", "name": "Hviletid", "type": "numeric"},
        ],
        editable=True,
        row_deletable=True,
        page_current=0,
        page_size=10,
        dropdown={
            "type": {"options": [{"label": i.name, "value": i.id} for i in types]},
            "fuel": {"options": [{"label": i.name, "value": i.id} for i in fuel]},
            "leasing_type": {
                "options": [{"label": i.name, "value": i.id} for i in leasingtypes]
            },
            "location": {
                "options": [{"label": i.address, "value": i.id} for i in locations]
            },
        },
    )

    return configuration_table


def init_new_car_modal():

    with Session() as s:
        types = s.query(VehicleTypes).all()
        fuel = s.query(FuelTypes).all()
        leasingtypes = s.query(LeasingTypes).all()
        locations = s.query(AllowedStarts).all()

    modal = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Tilføj køretøj")),
                    dbc.ModalBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Nummerplade"),
                                            dbc.Input(id="plate_in", type="text"),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Mærke"),
                                            dbc.Input(id="make_in", type="text"),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Model"),
                                            dbc.Input(id="model_in", type="text"),
                                        ]
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Køretøjstype"),
                                            dcc.Dropdown(
                                                [
                                                    {"label": i.name, "value": i.id,}
                                                    for i in types
                                                ],
                                                id="type_in",
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Drivmiddel"),
                                            dcc.Dropdown(
                                                [
                                                    {"label": i.name, "value": i.id,}
                                                    for i in fuel
                                                ],
                                                id="fuel_in",
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("wltp_fosil"),
                                            dbc.Input(
                                                id="wltp_fossil_in", type="number"
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("wltp_el"),
                                            dbc.Input(id="wltp_el_in", type="number"),
                                        ]
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Procentvis wltp nedskrivning"),
                                            dbc.Input(
                                                id="capacity_decrease_in", type="number"
                                            ),
                                            dbc.FormFeedback(
                                                "Nedskrivning skal være et tal imellem 0 og 100",
                                                type="invalid",
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("co2 pr km"),
                                            dbc.Input(id="co2_pr_km_in", type="number"),
                                        ]
                                    ),
                                ]
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Rækkevidde"),
                                            dbc.Input(id="range_in", type="number"),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("Omkostning / år"),
                                            dbc.Input(
                                                id="omkostning_aar_in", type="number"
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            dbc.Label("Lokation"),
                            dcc.Dropdown(
                                [
                                    {"label": i.address, "value": i.id}
                                    for i in locations
                                ],
                                id="location_in",
                            ),
                            dbc.Label("Leasingperiode"),
                            html.Br(),
                            dcc.DatePickerRange(
                                id="leasing_period_in",
                                initial_visible_month=date(2017, 8, 5),
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            dbc.Label("Leasingtype"),
                                            dcc.Dropdown(
                                                [
                                                    {"label": i.name, "value": i.id,}
                                                    for i in leasingtypes
                                                ],
                                                id="leasing_type_in",
                                            ),
                                        ]
                                    ),
                                    dbc.Col(
                                        [
                                            dbc.Label("km / år"),
                                            dbc.Input(id="km_aar_in", type="numeric"),
                                        ]
                                    ),
                                ]
                            ),
                        ]
                    ),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Tilføj køretøj",
                            color="primary",
                            id="add",
                            className="ms-auto",
                            n_clicks=0,
                        )
                    ),
                ],
                id="modal",
                is_open=False,
            ),
        ]
    )

    return modal


description_header = dbc.Toast(
    [
        html.P(
            "På denne side kan man rette i oplysninger om indregistrerede køretøjer og tilføje nye køretøjer man"
            " gerne vil bruge i simuleringsværktøjer. Vær observant på at kommatal skal skrives i engelsk format, dvs. punktum i stedet for komma.",
            className="mb-0",
        )
    ],
    header="Konfiguration",
    className="description-header",
)

layout = html.Div(
    [
        description_header,
        html.Div(
            [],
            # for warnings
            id="alerts",
        ),
        html.H3("Flådeoverblik"),
        html.Div(
            [
                html.Div(
                    [
                        html.A(
                            [
                                html.I(
                                    className="fas fa-download",
                                    style={"vertical-align": "middle"},
                                ),
                                html.Span(
                                    "Download til .xlsx", style={"margin-left": "5px"}
                                ),
                            ],
                            id="download_button",
                            download="True",
                            className="download-link",
                        ),
                    ],
                    className="flex-container",
                ),
                dcc.Download(id="download-dataframe-xlsx"),
                init_table(),
            ]
        ),
        html.Div(
            [
                dbc.Button("Gem ændringer", color="primary", id="update_btn"),
                dbc.Button("Tilføj køretøj", color="primary", id="add_vehicle"),
            ],
            className="flex-container",
        ),
        init_new_car_modal(),
        html.Div(id="table_dummy"),
    ],
    className="table_container",
)


@callback(
    Output("capacity_decrease_in", "invalid"), Input("capacity_decrease_in", "value"),
)
def validate_capacity_decrease_is_percent(value):
    if value is None:
        return False
    elif value >= 0 and value <= 100:
        return False
    else:
        return True


@callback(
    Output("download-dataframe-xlsx", "data"),
    Input("download_button", "n_clicks"),
    prevent_initial_call=True,
)
def download_cars_data(n_clicks):
    df = pd.read_sql(
        Query(
            [
                data_access.dbschema.Cars,
                VehicleTypes.name.label("Type"),
                FuelTypes.name.label("Drivmiddel"),
                LeasingTypes.name.label("Leasingtype"),
                AllowedStarts.address.label("Adresse"),
            ]
        )
        .join(VehicleTypes)
        .join(FuelTypes)
        .join(LeasingTypes)
        .join(AllowedStarts)
        .statement,
        engine,
    )

    df = df.drop(columns=["type", "fuel", "leasing_type", "location"])
    df = df.rename(
        columns={
            "plate": "Nummnerplade",
            "make": "Mærke",
            "model": "Model",
            "wltp_fossil": "Wltp (fossil)",
            "wltp_el": "Wltp (el)",
            "capacity_decrease": "Procentvis wltp",
            "co2_pr_km": "CO2 (g/km)",
            "range": "Rækkevidde (km)",
            "omkostning_aar": "Omk./år",
            "start_leasing": "Start leasing",
            "end_leasing": "Slut leasing",
        }
    )
    return dcc.send_data_frame(df.to_excel, "flåde.xlsx")


def add_vehicle(
    plate_in,
    make_in,
    model_in,
    type_in,
    fuel_in,
    wltp_fossil_in,
    wltp_el_in,
    capacity_decrease_in,
    co2_pr_km_in,
    range_in,
    omkostning_aar_in,
    location_in,
    start_leasing_in,
    end_leasing_in,
    leasing_type_in,
    km_aar_in,
):
    def float_to_none(f):
        return f or None

    try:
        with Session() as sess:
            c = Cars(
                plate=plate_in,
                make=make_in,
                model=model_in,
                type=type_in,
                fuel=fuel_in,
                wltp_fossil=float_to_none(wltp_fossil_in),
                wltp_el=float_to_none(wltp_el_in),
                capacity_decrease=float_to_none(capacity_decrease_in),
                co2_pr_km=float_to_none(co2_pr_km_in),
                range=float_to_none(range_in),
                omkostning_aar=float_to_none(omkostning_aar_in),
                location=location_in,
                start_leasing=None
                if start_leasing_in is None
                else datetime.strptime(start_leasing_in, "%Y-%m-%d"),
                end_leasing=None
                if end_leasing_in is None
                else datetime.strptime(end_leasing_in, "%Y-%m-%d"),
                leasing_type=leasing_type_in,
                km_aar=float_to_none(km_aar_in),
            )
            sess.add(c)
            sess.commit()
    except Exception as e:
        return error_alert(str(e)), get_cars_rounded()
    return success_alert(), get_cars_rounded()


def success_alert():
    return dbc.Toast(
        "Konfigurationen blev opdateret uden problemer",
        id="sql-success",
        header="Success",
        is_open=True,
        duration=4000,
        dismissable=True,
        icon="success",
        className="data-alert",
    )


def error_alert(error):
    return dbc.Toast(
        error,
        id="sql-success",
        header="Fejl",
        is_open=True,
        dismissable=True,
        icon="danger",
        className="data-alert",
    )


def warning_alert(warning):
    return dbc.Toast(
        warning,
        id="sql-success",
        header="Advarsel",
        is_open=True,
        dismissable=True,
        icon="warning",
        className="data-alert",
    )


@callback(
    Output("alerts", "children"),
    Output("config_table", "data"),
    Input("update_btn", "n_clicks"),
    Input("add", "n_clicks"),
    State("plate_in", "value"),
    State("make_in", "value"),
    State("model_in", "value"),
    State("type_in", "value"),
    State("fuel_in", "value"),
    State("wltp_fossil_in", "value"),
    State("wltp_el_in", "value"),
    State("capacity_decrease_in", "value"),
    State("co2_pr_km_in", "value"),
    State("range_in", "value"),
    State("omkostning_aar_in", "value"),
    State("location_in", "value"),
    State("leasing_period_in", "start_date"),
    State("leasing_period_in", "end_date"),
    State("leasing_type_in", "value"),
    State("km_aar_in", "value"),
)
def update_db(
    ub,
    ab,
    plate_in,
    make_in,
    model_in,
    type_in,
    fuel_in,
    wltp_fossil_in,
    wltp_el_in,
    capacity_decrease_in,
    co2_pr_km_in,
    range_in,
    omkostning_aar_in,
    location_in,
    start_leasing_in,
    end_leasing_in,
    leasing_type_in,
    km_aar_in,
):
    ctx = dash.callback_context
    if ctx.triggered[0].get("prop_id") == "update_btn.n_clicks":
        return update_cars()
    elif ctx.triggered[0].get("prop_id") == "add.n_clicks":
        return add_vehicle(
            plate_in,
            make_in,
            model_in,
            type_in,
            fuel_in,
            wltp_fossil_in,
            wltp_el_in,
            capacity_decrease_in,
            co2_pr_km_in,
            range_in,
            omkostning_aar_in,
            location_in,
            start_leasing_in,
            end_leasing_in,
            leasing_type_in,
            km_aar_in,
        )
    else:
        return None, get_cars_rounded()


@callback(
    Output("modal", "is_open"),
    [Input("add_vehicle", "n_clicks"), Input("add", "n_clicks")],
    [State("modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open


deletions = []
changes = []


def update_cars():
    deleted_updates = []
    try:
        with Session() as sess:
            sess.query(Cars).filter(Cars.id.in_(deletions)).delete(
                synchronize_session="fetch"
            )
            # Find out if user is trying to update items that have been deleted by other users
            for item in changes:
                r = (
                    sess.query(Cars)
                    .filter(Cars.id == item["id"])
                    .update({key: val for key, val in item.items() if val})
                )
                if r == 0:
                    deleted_updates.append(str(item.get("id")))
            sess.commit()
    except Exception as e:
        return error_alert(str(e)), get_cars_rounded()
    if len(deleted_updates) == 0:
        return success_alert(), get_cars_rounded()
    else:
        warning_string = (
            f'Opdateringerne til køretøjeret/køretøjerne med id {", ".join(deleted_updates)} '
            f"er ikke påført da de er blevet slettet af en anden bruger"
        )
        return warning_alert(warning_string), get_cars_rounded()


@callback(
    Output("table_dummy", "children"),
    Input("config_table", "data"),
    Input("config_table", "data_previous"),
    prevent_initial_call=True,
)
def display_output(data, data_previous):
    # If update is triggered by a reload do nothing
    if data == data_previous:
        raise PreventUpdate
    # Dash doesn't have specific callbacks for different table changes so we have to resort to this
    if data_previous is not None and len(data) < len(data_previous):
        deleted_id = [item for item in data_previous if item not in data][0].get("id")
        deletions.append(deleted_id)
        for i in range(len(changes)):
            if changes[i].get("id") == deleted_id:
                del changes[i]
    if data_previous is not None and len(data) == len(data_previous):
        print([item for item in data if item not in data_previous])
        changed_item = [item for item in data if item not in data_previous][0]
        for s in ("start_leasing", "end_leasing"):
            if s in changed_item:
                changed_item[s] = datetime.strptime(changed_item[s], "%Y-%m-%d")
        for i in range(len(changes)):
            if changes[i].get("id") == changed_item.get("id"):
                changes[i] = changed_item
                return None
        changes.append(changed_item)
    return None
