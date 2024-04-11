import dash_bootstrap_components as dbc


def view_tabs_layout():
    markup = [
        dbc.Tab('', label="Model View"),
        dbc.Tab('', label="Dataset View"),
    ]
    return dbc.Tabs(markup)
