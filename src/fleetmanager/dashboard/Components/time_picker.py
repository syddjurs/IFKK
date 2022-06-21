import json
import re
import uuid

import dash
import dash_bootstrap_components as dbc
from dash import (
    ALL,
    MATCH,
    Dash,
    Input,
    Output,
    State,
    callback,
    callback_context,
    dash,
    dcc,
    html,
)


class MultiTimePickerAIO(html.Div):
    class ids:
        add_button = lambda aio_id: {
            "component": "MultiTimePickerAIO",
            "subcomponent": "AddButton",
            "aio_id": aio_id,
        }
        time_selectors = lambda aio_id: {
            "component": "MultiTimePickerAIO",
            "subcomponent": "div",
            "aio_id": aio_id,
        }

    ids = ids

    def __init__(self, title, aio_id=None):

        if aio_id is None:
            self.aio_id = str(uuid.uuid4())

        row_id = str(uuid.uuid4())

        super().__init__(
            [
                dbc.Label(title),
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label("Start"),
                                        dbc.Input(
                                            type="text",
                                            placeholder="08:00",
                                            value="08:00",
                                            id={
                                                "component": "MultiTimePickerAIO",
                                                "type": "bike_start",
                                                "id": row_id,
                                            },
                                        ),
                                        dbc.FormFeedback(
                                            "Tid skal være i formatet HH:MM",
                                            type="invalid",
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label("Slut"),
                                        dbc.Input(
                                            type="text",
                                            placeholder="18:00",
                                            value="18:00",
                                            id={
                                                "component": "MultiTimePickerAIO",
                                                "type": "bike_end",
                                                "id": row_id,
                                            },
                                        ),
                                        dbc.FormFeedback(
                                            "Tid skal være i formatet HH:MM",
                                            type="invalid",
                                        ),
                                    ]
                                ),
                                dbc.Col(
                                    [
                                        dbc.Button(
                                            "Slet",
                                            color="primary",
                                            className="delete-button",
                                            id={
                                                "component": "MultiTimePickerAIO",
                                                "subcomponent": "delete_bike",
                                                "id": row_id,
                                            },
                                        )
                                    ]
                                ),
                            ],
                            id={
                                "component": "MultiTimePickerAIO",
                                "subcomponent": "bike_row",
                                "id": row_id,
                            },
                        ),
                    ],
                    id=self.ids.time_selectors(self.aio_id),
                ),
                dbc.Button(
                    "Tilføj tidsinterval",
                    style={"margin-top": "10px"},
                    color="primary",
                    id=self.ids.add_button(self.aio_id),
                ),
            ]
        )

    @callback(
        Output(
            {"component": "MultiTimePickerAIO", "type": "bike_start", "id": ALL},
            "invalid",
        ),
        Output(
            {"component": "MultiTimePickerAIO", "type": "bike_end", "id": ALL},
            "invalid",
        ),
        Input(
            {"component": "MultiTimePickerAIO", "type": "bike_start", "id": ALL},
            "value",
        ),
        Input(
            {"component": "MultiTimePickerAIO", "type": "bike_end", "id": ALL}, "value"
        ),
    )
    def validate_timestamps(values_start, values_end):
        def timevalidation(time):
            time_regex = "^([0-9]|0[0-9]|1[0-9]|2[0-3]):[0-5][0-9]$"
            if time is None:
                return True
            if re.search(time_regex, time) is None:
                return True
            else:
                return False

        start_mask = list(map(lambda time: timevalidation(time), values_start))
        end_mask = list(map(lambda time: timevalidation(time), values_end))
        return start_mask, end_mask

    @callback(
        Output(ids.time_selectors(MATCH), "children"),
        Input(ids.add_button(MATCH), "n_clicks"),
        Input(
            {
                "component": "MultiTimePickerAIO",
                "subcomponent": "delete_bike",
                "id": ALL,
            },
            "n_clicks",
        ),
        State(ids.time_selectors(MATCH), "children"),
        prevent_initial_call=True,
    )
    def add_time_selector(add_click, delete_click, children):
        ctx = callback_context
        id_dict = json.loads(ctx.triggered[0]["prop_id"].split(".")[0])
        if id_dict["subcomponent"] == "AddButton":
            row_id = str(uuid.uuid4())
            return children + [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Label("Start"),
                                dbc.Input(
                                    type="text",
                                    placeholder="08:00",
                                    id={
                                        "component": "MultiTimePickerAIO",
                                        "type": "bike_start",
                                        "id": row_id,
                                    },
                                ),
                                dbc.FormFeedback(
                                    "Tid skal være i formatet HH:MM", type="invalid",
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Label("Slut"),
                                dbc.Input(
                                    type="text",
                                    placeholder="18:00",
                                    id={
                                        "component": "MultiTimePickerAIO",
                                        "type": "bike_end",
                                        "id": row_id,
                                    },
                                ),
                                dbc.FormFeedback(
                                    "Tid skal være i formatet HH:MM", type="invalid",
                                ),
                            ]
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    "Slet",
                                    color="primary",
                                    className="delete-button",
                                    id={
                                        "component": "MultiTimePickerAIO",
                                        "subcomponent": "delete_bike",
                                        "id": row_id,
                                    },
                                )
                            ]
                        ),
                    ],
                    id={
                        "component": "MultiTimePickerAIO",
                        "subcomponent": "bike_row",
                        "id": row_id,
                    },
                )
            ]
        else:
            return list(
                filter(lambda row: row["props"]["id"]["id"] != id_dict["id"], children)
            )
