from dash import Dash, dcc, html, Input, Output, dash_table, callback, dependencies
from dash.exceptions import PreventUpdate
import dash
from time import sleep
from random import randint, seed
import pandas as pd
import dash_bootstrap_components as dbc

df = pd.read_csv('data.csv')


# For the documentation to always render the same values
def generate_table(df):
    return dash_table.DataTable(
    id = 'data-table',
    data=df.to_dict('records'),
    columns=[{'id': c, 'name': c} for c in df.columns],
    style_cell={
                'font-family': 'montserrat',
                'textAlign': 'center',
                'border': 'none'},
    editable=True,
#    style_cell_conditional=[
#        {
#            'if': {'column_id': 'Region'},
#            'textAlign': 'center'
#        }
#    ]
)

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

style_txt = {'font-size': '32px', 
             'font-family': 'montserrat', 
             'textAlign': 'center'}

app.layout = html.Div(
    [
        dbc.Row(
            dbc.Col(
                dcc.Dropdown(
                    id='dropdown',
                    options=[{'label': i, 'value': i} for i in list(df.columns)],
                    multi=True, placeholder='Select a column',
                    style={'backgroundColor': 'white',
                        'width': '25em',
                        'border-radius': '1em',
                        'margin-bottom': '3em',
                        'margin-top': '3em',
                        'margin-left': '1em'}
                ), style={'allign': 'left'}
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.Div('Id'), 
                    width={"size": 1, "order": 1},
                    style=style_txt
                ),
                dbc.Col(
                    html.Div('Features'),
                    width={"order": 2, "offset": 2},
                    style=style_txt
                ),
                dbc.Col(
                    html.Div('Score'),
                    width={"order": 3, "offset": 2},
                    style=style_txt
                ),
                dbc.Col(
                    html.Div('Prediction'),
                    width={"order": 4},
                    style=style_txt
                ),
                dbc.Col(
                    html.Div('Actual'),
                    width={"order": 5},
                    style=style_txt
                )
            ], 
            style={
                'margin-left': '3em',
                'margin-right': '3em'
            }
        ),
        dbc.Row(dbc.Col(html.Div(id='table-container')))
    ]
)


@app.callback(dash.dependencies.Output('table-container', 'children'),
    [dash.dependencies.Input('dropdown', 'value')])

def display_table(dropdown_value):
    if dropdown_value is None:
        return generate_table(df)

    ## add an 'or' condition for the other column you want to use to slice the df 
    ## and update the columns that are displayed
    dff = pd.DataFrame(df, columns = dropdown_value)
    return generate_table(dff)

if __name__ == '__main__':
    app.run_server(debug=True, port=8081)