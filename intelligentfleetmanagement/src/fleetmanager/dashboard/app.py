import dash_bootstrap_components as dbc
import diskcache
from dash import Dash
from dash.long_callback import DiskcacheLongCallbackManager

THEME = "bootstrap"
dbc_css = (
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.2/dbc.min.css"
)

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

app = Dash(
    __name__,
    external_stylesheets=[
        getattr(dbc.themes, THEME.upper()),
        dbc_css,
        {
            "href": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css",
            "rel": "stylesheet",
            "crossorigin": "anonymous",
        },
    ],
    long_callback_manager=long_callback_manager,
    suppress_callback_exceptions=True,
)
