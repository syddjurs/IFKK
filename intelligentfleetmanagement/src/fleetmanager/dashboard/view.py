from fnmatch import fnmatch

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash_bootstrap_templates import load_figure_template

from .app import THEME, app

load_figure_template(THEME.lower())

from .page_config import layout as config_layout
from .page_fleet import layout as fleet_layout
from .page_goal import layout as goal_layout
from .page_setup import layout as setup_layout

sidebar = html.Div(
    [
        html.H2("Simulerings-\ntool"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink(
                    [html.I(className="fas fa-list"), "Simuleringssetup"],
                    href="/",
                    active="exact",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-car"), "Flådesammensætning"],
                    href="/page_fleet",
                    active="exact",
                    disabled=True,
                    id="fleet_link",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-microchip"), "Målsimulering"],
                    href="/page_goal",
                    active="exact",
                    disabled=True,
                    id="sim_link",
                ),
                dbc.NavLink(
                    [html.I(className="fas fa-user-shield"), "Konfiguration"],
                    href="/page_config",
                    active="exact",
                    className="config-nav",
                ),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    className="sidebar",
)
content = html.Div(id="page-content", className="content")

fig = go.Figure()
layout = dbc.Container(
    [
        dcc.Location(id="url"),
        sidebar,
        content,
        html.Div(id="dummy"),
        dcc.Store(id="trip_store"),
        dcc.Store(id="location_store"),
        dcc.Store(id="location_name"),
        dcc.Store(id="vehicle_idx_store"),
        dcc.Store(id="vehicle_sel_store"),
        dcc.Store(id="fleet_store"),
        dcc.Store(id={"type": "fig_store", "index": 1}, data=fig),
        dcc.Store(id={"type": "fig_store", "index": 2}, data=fig),
        dcc.Store(id={"type": "fig_store", "index": 3}, data=fig),
        dcc.Store(id={"type": "goal_fig_store", "index": 1}, data=fig),
        dcc.Store(id={"type": "goal_fig_store", "index": 2}, data=fig),
        dcc.Store(id="date_store"),
        dcc.Store(id="sim_daterange_store"),
        dcc.Store(id="trips_without_vehicle_store"),
        dcc.Store(id="savings_pr_year_store"),
        dcc.Store(id="savings_pr_year_co2e_store"),
    ],
    className="dbc",
)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page_content(pathname):
    if pathname == "/" or pathname == "" or fnmatch(pathname, "/tool/*/dash/*"):
        return setup_layout
    elif pathname == "/page_fleet":
        return fleet_layout
    elif pathname == "/page_goal":
        return goal_layout
    elif pathname == "/page_config":
        return config_layout

    return (
        html.Div(
            [
                html.H1("404: Not found", className="text-danger"),
                html.Hr(),
                html.P(f"The pathname {pathname} was not recognised..."),
            ]
        ),
        False,
    )


app.layout = layout
app.validation_layout = html.Div([layout, setup_layout, fleet_layout, config_layout])
server = app.server
