import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

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
                    disabled=False,
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
    ],
    className="dbc",
)


@callback(
    [Output("page-content", "children"), Output("fleet_link", "disabled")],
    Input("url", "pathname"),
    State("fleet_link", "disabled"),
)
def render_page_content(pathname, fleet_page):
    if pathname == "/" or pathname == "":
        return (setup_layout, fleet_page)
    elif pathname == "/page_fleet":
        return (fleet_layout, False)
    elif pathname == "/page_goal":
        return (goal_layout, False)
    elif pathname == "/page_config":
        return (config_layout, fleet_page)

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
