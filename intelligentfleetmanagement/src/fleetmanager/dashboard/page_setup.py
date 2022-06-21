from datetime import datetime

import dash_bootstrap_components as dbc
from dash import ALL, Input, Output, State, callback, callback_context, dcc, html
from dash.exceptions import PreventUpdate

import fleetmanager.data_access.db_engine as db
from .utils import get_emission, toggle_all
from fleetmanager.dashboard.utils import input_table
from fleetmanager.data_access import AllowedStarts, Cars

location_view = [
    html.H3("Lokationer"),
    dbc.Card(
        [
            dbc.CardBody(
                [
                    dbc.Row([dbc.Col(html.H4("Adresse"))], className="header-row"),
                    dbc.RadioItems(
                        id="location_view",
                    ),
                ]
            )
        ]
    ),
]

datepicker_view = dbc.Card(
    dcc.DatePickerRange(
        start_date="2022-01-20",
        end_date="2022-03-01",
        persistence=True,
        id="location_date",
        max_date_allowed=datetime.now(),
        className="location-date-picker center-content",
    ),
)

vehicle_view = [
    html.H3("Køretøjer tilknyttet til den valgte lokation"),
    dbc.Card(
        [
            dbc.CardBody(
                [
                    html.Div(
                        id="vehicle_view",
                        className="vehicle-card",
                    )
                ]
            )
        ],
        className="bot-margin",
    ),
    dbc.Button(
        "Vælg alle",
        color="success",
        className="left-btn",
        id="choose_all",
        style={"visibility": "hidden"}
    ),
    dcc.Store("choose_value_holder"),
    dbc.Nav(
        [
            dbc.NavItem(
                dbc.NavLink(
                    "Flådesammensætning",
                    href="/page_fleet",
                    active="exact",
                    className="btn btn-primary",
                    disabled=True,
                    id='pf_link'
                )
            ),
            dbc.NavItem(
                dbc.NavLink(
                    "Målsimulering",
                    href="/page_goal",
                    active="exact",
                    className="btn btn-primary",
                    disabled=True,
                    id="pg_link"
                )
            ),
        ],
        className="right-btn",
    ),
]

description_header = dbc.Toast(
    [
        html.P(
            "Her udvælges lokationer, tidsperiode samt køretøjer, der skal danne baggrund for simuleringerne."
            " Når denne side er udfyldt, så kan både flådesammensætning of målsimulering tilgåes",
            className="mb-0",
        )
    ],
    header="Simuleringssetup",
    className="description-header",
)

layout = html.Div(
    [
        description_header,
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row(dbc.Col(location_view), className="location_card"),
                        html.Br(),
                        dbc.Row(dbc.Col(datepicker_view)),
                    ],
                    width="4",
                ),
                dbc.Col(vehicle_view),
            ],
        ),
    ],
)


Session = db.session_factory(db.engine_creator())


@callback(
    Output("location_view", "options"),
    Output("date_store", "data"),
    Input("location_date", "start_date"),
    Input("location_date", "end_date"),
)
def location_view(start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    with Session() as session:
        rows = (
            session.query(Cars, AllowedStarts.address, AllowedStarts.id.label("loc_id"))
            .join(AllowedStarts)
            .filter(Cars.omkostning_aar.isnot(None))
        )
        locations = list(set([(row.loc_id, row.address) for row in rows]))
    sorted_list = sorted([a[1] for a in locations])
    locations = {a[1]: a[0] for a in locations}
    return [{"label": label, "value": locations[label]} for label in sorted_list], [
        start,
        end,
    ]


@callback(
    Output("vehicle_view", "children"),
    Output({"type": "vehicle_checklist", "index": ALL}, "value"),
    Output("location_store", "data"),
    Output("location_name", "data"),
    Output("choose_all", "style"),
    Output("choose_value_holder", "data"),
    Output("fleet_link", "disabled"),
    Output("sim_link", "disabled"),
    Output("pf_link", "disabled"),
    Output("pg_link", "disabled"),
    Input("location_view", "value"),
    Input("location_date", "start_date"),
    Input("location_date", "end_date"),
    Input("choose_all", "n_clicks"),
    Input("choose_value_holder", "data"),
    State({"type": "vehicle_checklist", "index": ALL}, "value"),
    State("vehicle_view", "children"),
    State("location_name", "data"),
)
def vehicle_view(
    location_id,
    start_date,
    end_date,
    choose_all,
    choose_all_value,
    selected_vehicles,
    vw,
    lm,
):
    if location_id is None:
        raise PreventUpdate
    if choose_all_value is None or choose_all is None:
        pass
    elif choose_all_value["location"] != location_id:
        pass
    else:
        if choose_all > choose_all_value["n_click"]:
            choose_all_value["n_click"] += 1
            n_click = choose_all_value["n_click"]
            choose_all = True if choose_all_value["selected"] is False else False
            choose_all_value = {
                "selected": choose_all,
                "n_click": n_click,
                "location": location_id,
            }
            changed = toggle_all(vw, choose_all)
            return (
                changed,
                [choose_all] * len(selected_vehicles),
                location_id,
                lm,
                {"visibility": "visible", "float": "left"},
                choose_all_value,
                False,
                False,
                False,
                False,
            )
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    choose_all = True
    n_click = 0
    with Session() as session:
        location_name = (
            session.query(AllowedStarts.address).filter(AllowedStarts.id == location_id)
        )[0]
        vehicles = (
            session.query(
                Cars.id,
                Cars.plate,
                Cars.make,
                Cars.model,
                Cars.omkostning_aar,
                Cars.co2_pr_km,
                AllowedStarts,
                Cars.wltp_el,
                Cars.wltp_fossil,
            )
            .filter(
                (AllowedStarts.id == location_id)
                & ((Cars.end_leasing >= end) | (Cars.end_leasing.is_(None)))
                & (Cars.omkostning_aar.isnot(None))
            )
            .join(AllowedStarts)
        )

        eligible_vehicles = []
        for vehicle in vehicles:
            eligible_vehicles.append(
                [
                    vehicle.id,
                    "{} {} {}".format(vehicle.plate, vehicle.make, vehicle.model),
                    vehicle.omkostning_aar,
                    get_emission(vehicle),
                ]
            )

    return (
        input_table(
            eligible_vehicles,
            header=[
                "Biltype",
                "Årlige Omkostninger (DKK)",
                "Drivmiddel forbrug",
                "Medtag i simulation",
            ],
            id="vehicle_checklist",
            fn_input=dbc.Checkbox,
            fn_input_value=choose_all,
        ),
        [choose_all] * len(selected_vehicles),
        location_id,
        location_name,
        {"visibility": "visible", "float": "left"},
        {"selected": choose_all, "n_click": n_click, "location": location_id},
        False,
        False,
        False,
        False,
    )


@callback(
    Output("vehicle_sel_store", "data"),
    Input({"type": "vehicle_checklist", "index": ALL}, "name"),
    Input({"type": "vehicle_checklist", "index": ALL}, "value"),
    prevent_initial_call=True,
)
def vehicle_selection(id_checklist, checklist):
    ctx = callback_context
    if checklist is None or not ctx.triggered:
        raise PreventUpdate

    return [i for i, check in zip(id_checklist, checklist) if check]
