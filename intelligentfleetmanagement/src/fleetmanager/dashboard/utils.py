from itertools import cycle
import time

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import html


def card(body, header=None, style=None):
    card_body = [dbc.CardBody(body)]
    if header is not None:
        card_body.insert(0, dbc.CardHeader(header))
    return dbc.Card(card_body)


def input_table(
    data,
    header=None,
    id=None,
    col_style=None,
    fn_input=None,
    fn_params=None,
    total_id=None,
    fn_input_value=False
):
    if header is None:
        table_header = None
    else:
        table_header = html.Thead(html.Tr([html.Th(h) for h in header]))
    if fn_params is None:
        fn_params = {}
    if col_style is None:
        col_style = cycle([{}])

    rows = []
    total0 = 0
    for i, (r, cstyle) in enumerate(zip(data, col_style)):
        row = []
        for c in r[1:]:
            row.append(html.Td(c, style=cstyle))
        try:
            total0 += int(r[-1])
        except (ValueError, TypeError):
            pass

        if fn_input is not None:
            params = {
                "id": {"type": id, "index": i},
                # Name isn't used for anything else so we use it to store id's
                "name": r[0],
                "value": fn_input_value,
                "persistence": True,
            }
            params.update(fn_params)
            row.append(
                html.Td(
                    fn_input(
                        **params,
                    )
                )
            )
        rows.append(html.Tr(row))
    table = [table_header, html.Tbody(rows)]
    if total_id is not None:
        table.append(
            html.Thead(
                html.Tr(
                    [html.Td(html.B("Total"))]
                    + [html.Td()] * (len(r) - 3)
                    + [html.Td(html.B(total0)), html.Td(html.B(id=total_id))]
                )
            )
        )
    return dbc.Table(table)


def distance_histogram_from(source, xlabel=None, ylabel=None, title=None):
    bins = source.pop("edges")
    bins = 0.5 * (bins[:-1] + bins[1:])
    fig = go.Figure()
    for color, (key, val) in zip(
        ["#52575C", "#FF6760", "#109CF1", "#FFBC1F", "#40DD7F"], source.items()
    ):
        fig.add_bar(x=bins, y=val, name=key, marker_color=color)
    fig.update_layout(
        bargap=0.0,
        barmode="stack",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        title_text=title,
    )
    return fig


def capacity_plot_from(source, xlabel=None, ylabel=None, title=None):
    fig = go.Figure()
    for color, name, key in zip(
        ["#109CF1", "#FF6760"],
        ["Nuværende", "Simulation"],
        ["cur_unassigned_trips", "sim_unassigned_trips"],
    ):
        fig.add_scatter(
            x=source.data["timeframe"],
            y=source.data[key],
            name=name,
            line={"color": color},
        )
    fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title_text=title)
    fig.update_yaxes(rangemode="nonnegative", dtick=1)
    return fig


def consequence_table_from(source):
    table_header = [
        html.Tr(
            [
                html.Th("Konsekvens"),
                html.Th("Nuværende værdi"),
                html.Th("Simuleret værdi"),
            ]
        )
    ]
    rows = []
    for col1, col2, col3 in zip(
        source.data["keys"], source.data["cur_values"], source.data["sim_values"]
    ):
        rows.append(
            html.Tr(
                [
                    html.Td(col1),
                    html.Td("{:.2f}".format(col2)),
                    html.Td("{:.2f}".format(col3)),
                ]
            )
        )
    table_body = [html.Tbody(rows)]
    return table_header + table_body


def prepare_trip_store(trips, location_name):
    # storing the trips
    save = trips[["start_time", "end_time", "distance", "current", "simulation"]].copy()
    save["Nuværende"] = save.current.apply(
        lambda x: f"{x.make} {x.model} {x.name.split('_')[-1]} {x.plate}"
        if x.name != "Unassigned"
        else "Ikke allokeret"
    )
    save["Simulering"] = save.simulation.apply(
        lambda x: f"{x.make} {x.model} {x.name.split('_')[-1]} {x.plate}"
        if x.name != "Unassigned"
        else "Ikke allokeret"
    )
    save.drop(["current", "simulation"], axis=1, inplace=True)
    save["Adresse"] = [location_name[0]] * len(save)
    trip_store = save.to_dict("records")  # pickle.dumps(save)
    return trip_store


def get_emission(entry):
    if any(
        [
            (entry.wltp_fossil == 0 and entry.wltp_el == 0),
            (pd.isna(entry.wltp_fossil) and pd.isna(entry.wltp_el)),
        ]
    ):
        return "0"

    udledning = (
        f"{str(round(entry.wltp_el)).replace('.', ',')} Wh/km"
        if pd.isna(entry.wltp_fossil) or entry.wltp_fossil == 0
        else f"{str(round(entry.wltp_fossil, 1)).replace('.', '.')} km/l"
    )
    return udledning


def goal_bar_plot(keys, values, title=None, xlabel=None, ylabel=None):
    fig = go.Figure([go.Bar(x=keys, y=values)])
    fig.update_layout(title_text=title, title_x=.5, xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def unix_time_millis(dt):
    """Convert datetime to unix timestamp"""
    return int(time.mktime(dt.timetuple()))


def unix_to_datetime(unix):
    """Convert unix timestamp to datetime."""
    return pd.to_datetime(unix, unit="s")


def accordian_table(fleet_header, fleet, current_expense, simulation_expense, savings, current_emission, simulation_emission, emission_savings):
    accordian = dbc.Row(
        [
            dbc.Col(
                [
                    html.B("Flådesammensætning"),
                    dbc.Table(
                        [fleet_header, html.Tbody(fleet)],
                    ),
                ],
                className="solution-fleet",
            ),
            dbc.Col(
                [
                    html.B("Simuleringsdetaljer"),
                    dbc.Table(
                        [
                            html.Thead(
                                html.Tr(
                                    [
                                        html.Th("Type"),
                                        html.Th("Enhed"),
                                    ]
                                )
                            ),
                            html.Tbody(
                                [
                                    html.Tr(
                                        [
                                            html.Td("Nuværende total omkostning"),
                                            html.Td(
                                                f"{round(current_expense)} DKK/år"
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Simuleret total omkostning"),
                                            html.Td(
                                                f"{round(simulation_expense)} DKK/år"
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td(
                                                html.B("Besparelse i omkostning")
                                            ),
                                            html.Td(html.B(f"{savings} DKK/år")),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Nuværende udledning"),
                                            html.Td(
                                                f"{str(round(current_emission, 3)).replace('.', ',')} Ton CO2e/år"
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td("Simuleret udledning"),
                                            html.Td(
                                                f"{str(round(simulation_emission, 3)).replace('.', ',')} Ton CO2e/år"
                                            ),
                                        ]
                                    ),
                                    html.Tr(
                                        [
                                            html.Td(
                                                html.B("Reduktion i udledning")
                                            ),
                                            html.Td(
                                                html.B(f"{emission_savings} Ton CO2e/år")
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                    ),
                ],
                className="solution-results",
            ),
        ]
    )
    return accordian


def toggle_all(vehicle_view, change_to=True):
    copied = vehicle_view.copy()
    for vehicle in copied['props']['children'][-1]['props']['children']:
        vehicle['props']['children'][-1]['props']['children']['props']['value'] = change_to
    return copied
