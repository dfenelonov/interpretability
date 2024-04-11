from dash import html

from interpretability.components.feature_list import FeatureListAIO
from interpretability.components.view_tabs import view_tabs_layout


def layout():
    """Page layout"""
    return html.Div(
        [
            view_tabs_layout(),
            FeatureListAIO("feature-list")
        ],
        id="main",
        className="container-fluid",
    )
