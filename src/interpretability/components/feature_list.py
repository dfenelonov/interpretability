import dash
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import callback, dcc, html, Input, Output, MATCH

from interpretability.data import df
from interpretability.analysis.explainer import explainer

AIO_NAME = "FeatureListAIO"


def build_id(subcomponent: str):
    """
    Creates lambda function which takes aio_id and returns subcomponent identifier as dictionary
    """
    return lambda aio_id: {"component": AIO_NAME, "subcomponent": subcomponent, "aio_id": aio_id}


class FeatureListAIO(html.Div):
    class ids:
        # @staticmethod
        # def row_id(aio_id, feature_name):
        #     return {"component": AIO_NAME, "aio_id": aio_id, "feature_name": feature_name}
        pass

    subcomponents = [
        "feature-list-container",
        "feature-filter-dropdown",
        "from-date-input",
        "to-date-input",
        "update-button",
        "identifier-input",
        "features-info",
        "markup-content"
    ]
    for subcomponent in subcomponents:
        setattr(ids, subcomponent.replace("-", "_"), build_id(subcomponent))

    @classmethod
    def get_markup(cls, aio_id):
        """Creates markup"""
        markup = [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label(
                                "Feature Filter",
                                html_for=cls.ids.feature_filter_dropdown(aio_id)
                            ),
                            dcc.Dropdown(
                                [
                                    {"label": column, "value": column} for column in df.columns
                                ],
                                id=cls.ids.feature_filter_dropdown(aio_id),
                                multi=True
                            )
                        ]
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                "From",
                                html_for=cls.ids.from_date_input(aio_id)
                            ),
                            dbc.Input(
                                type="date",
                                id=cls.ids.from_date_input(aio_id)
                            )
                        ]
                    ),
                    dbc.Col(
                        [
                            dbc.Label(
                                "To",
                                html_for=cls.ids.to_date_input(aio_id)
                            ),
                            dbc.Input(
                                type="date",
                                id=cls.ids.to_date_input(aio_id)
                            )
                        ]
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Update",
                            id=cls.ids.update_button(aio_id)
                        ),
                        align="end"
                    )
                ],
                class_name="p-2"
            ),
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Input(
                                    id=cls.ids.identifier_input(aio_id),
                                    placeholder="ID",
                                    type="number"
                                )
                            ),
                            dbc.Col(
                                "Score = ?"
                            )
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col("Features", style={"font-weight": "bold", "font-size": "large"}),
                            dbc.Col("Values", style={"font-weight": "bold", "font-size": "large"}),
                            dbc.Col("Impact", style={"font-weight": "bold", "font-size": "large"}),
                            dbc.Col("Percentile", style={"font-weight": "bold", "font-size": "large"})
                        ],
                        class_name="p-2"
                    ),
                    html.Div(
                        id=cls.ids.features_info(aio_id)
                    )
                ]
            )
        ]
        return html.Div(
            markup,
            id=cls.ids.markup_content(aio_id)
        )

    def __init__(self, aio_id=None):
        super(FeatureListAIO, self).__init__(self.get_markup(aio_id), id=self.ids.feature_list_container(aio_id))

    @staticmethod
    @callback(
        Output(ids.features_info(MATCH), "children"),
        Input(ids.feature_filter_dropdown(MATCH), "value"),
        Input(ids.identifier_input(MATCH), "value"),
        prevent_initial_call=True
    )
    def update_features_info(features, identifier):
        if features is None or identifier is None:
            return dash.no_update
        df_row = df.iloc[identifier].loc[features]
        total_rows = []
        for feature, feature_value in df_row.items():
            histogram_fig = go.Figure(go.Histogram(x=df[feature]))
            histogram_fig.update_xaxes(showticklabels=False)
            histogram_fig.update_layout(
                # autosize=True,
                width=200,
                height=200,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            shap_value = explainer.get_shap_values(identifier)[feature].values
            total_rows.append(
                dbc.Row(
                    [
                        dbc.Col(feature),
                        dbc.Col(feature_value),
                        dbc.Col(f"{shap_value:+.5f}", class_name="red" if shap_value < 0 else "green"),
                        dbc.Col(
                            dcc.Graph(
                                figure=go.Figure(data=histogram_fig),
                                config={"displayModeBar": False}
                            )
                        )
                    ]
                )
            )
        return total_rows
