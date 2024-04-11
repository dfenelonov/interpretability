import pathlib

import dash
import dash_bootstrap_components as dbc
from flask import send_from_directory

from interpretability.pages.explainability_page import layout as explainability_layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)


@app.server.route("/assets/<path:path>")
def static_file(path):
    """
    Load /assets/
    """
    static_folder = pathlib.Path(__file__).parent/"assets"
    return send_from_directory(static_folder, path)


app.layout = explainability_layout()


if __name__ == '__main__':
    app.run(debug=True)
