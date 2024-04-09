from dash import Dash, dcc, html, Input, Output, dash_table, callback, dependencies, State
from dash.exceptions import PreventUpdate
import dash
from time import sleep
from random import randint, seed
import pandas as pd
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from pandas.api.types import is_numeric_dtype
import joblib
from catboost import CatBoostRegressor

model = CatBoostRegressor() 

df = pd.read_csv('data.csv')

model.load_model('model_name')

df = df.round(2)



columnTypes = {
    "idxColumn": {"width": '195em'},
    "targetColumn" : {"width": '208em'}
}

value = 'Ok'

defaultColDef = {
    # set the default column width
    "width": 150,
    # make every column editable
    "editable": True
}

style_for_score = {"field": 'score',
                       "pinned": "right",
                        "type": "targetColumn",
                        "headerName": "",
                        "cellStyle": {'textAlign': 'center'}
                        }

style_for_descrete_score = {
                    "field": 'descrete_score',
                    "pinned": "right",
                    "type": "targetColumn",
                    'cellStyle': {
                        "function": "params.value == 'Approve' ? {'color': 'green', 'textAlign': 'center'} : {'color': 'red', 'textAlign': 'center'}"
                        },
                    "headerName": ""
                }

style_for_cat_target = {
                    "field": 'cat_target',
                    "pinned": "right",
                    "type": "targetColumn",
                    'cellStyle': {
                        "function": "params.value == 'Ok' ? {'color': 'green', 'textAlign': 'center'} : {'color': 'red', 'textAlign': 'center'}"
                        },
                    "headerName": ""
                    }


columnDefs = [{
    "field": 'Unnamed: 0',
    "type": "idxColumn",
    "headerName": "",
    "pinned": 'left',
    "cellStyle": {'textAlign': 'center'},
    "headerStyle": {'textAlign': 'center', 'font-size': '32px'}
    }]
columnDefs.extend([{"field": i, "cellStyle": {'textAlign': 'center'}, 'cellEditor': 'agNumberCellEditor'} if is_numeric_dtype(df[i]) else {"field": i, "cellStyle": {'textAlign': 'center'}} for i in list(df.columns[1:])])
columnDefs.append(style_for_score)
columnDefs.append(style_for_descrete_score)
columnDefs.append(style_for_cat_target)

# For the documentation to always render the same values
def generate_table(df):
    columnDefs = [{
                "pinned": "left",
                "field": 'Unnamed: 0',
                "type": "idxColumn",
                "cellStyle": {'textAlign': 'center'},
                "headerName": ""
                }]
    columnDefs.extend([{"field": i} for i in list(df.columns[1:-3])])
    columnDefs.append(style_for_score)
    columnDefs.append(style_for_descrete_score)
    columnDefs.append(style_for_cat_target)
    
    return [dag.AgGrid(
                        id ='editing-grid2',
                        columnDefs=columnDefs,
                        defaultColDef=defaultColDef,
                        rowData=df.to_dict("records"),
                        dashGridOptions={
                            'columnTypes': columnTypes,
                        },
                        persistence=True
                    )]

app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

style_txt = {'font-size': '32px', 
             'font-family': 'montserrat', 
             'textAlign': 'center'}

app.layout = html.Div(
    [
        dbc.Row(
            [
            dbc.Col(
                dcc.Dropdown(
                    id='dropdown',
                    options=[{'label': i, 'value': i} for i in list(set(df.columns)-set(['Unnamed: 0', 'score', 'descrete_score', 'cat_target']))],
                    multi=True, placeholder='Select a column',
                    style={'backgroundColor': 'white',
                        'width': '25em',
                        'border-radius': '1em',
                        'margin-bottom': '3em',
                        'margin-top': '3em',
                        'margin-left': '1em'}
                ), style={'allign': 'left'}
            ),
            dbc.Col(
                dbc.Button(
                    'Submit', 
                    size='sm', 
                    id='btn', 
                    title='Recalculate',
                    n_clicks=0,
                    style={'backgroundColor': 'grey',
                            'width': '25em',
                            'border-radius': '1em',
                            'margin-bottom': '3em',
                            'margin-top': '3em',
                            'margin-left': '1em'
                    }
                ), style={'allign': 'right'}
            )]
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
        dbc.Row(
            
                html.Div(
                    id = 'table-container',
                    children =
                        [
                            dag.AgGrid(
                                id ='editing-grid2',
                                columnDefs=columnDefs,
                                defaultColDef=defaultColDef,
                                rowData=df.to_dict("records"),
                                dashGridOptions={
                                    'columnTypes': columnTypes,
                                    "alignedGrids": ["bottom-grid-footer"]
                                },
                                persistence=True
                            )
                        ]
                )
            
        )
    ]
)

@app.callback(
    Output("editing-grid2", "rowData"),
    Input('btn', 'n_clicks'),
    State("editing-grid2", "cellValueChanged"),
    State("editing-grid2", "rowData"),
    prevent_initial_call=True)

def update(n, cell_val, row_data):
    if not cell_val:
        return df.to_dict("records")
    for row in cell_val:
        df.iloc[row['rowIndex']] = row['data']
    df['score'] = model.predict(df)
    return df.to_dict("records")

@app.callback(Output('table-container', 'children', allow_duplicate=True),
    [Input('dropdown', 'value')],
    prevent_initial_call=True)

def display_table(dropdown_value):
    if dropdown_value is None:
        return generate_table(df)
    if 'Unnamed: 0' not in dropdown_value:
        dropdown_value.insert(0, 'Unnamed: 0')
    dropdown_value.append('score')
    dropdown_value.append('descrete_score')
    dropdown_value.append('cat_target')
    ## add an 'or' condition for the other column you want to use to slice the df 
    ## and update the columns that are displayed
    dff = pd.DataFrame(df, columns = dropdown_value)
    return generate_table(dff)

if __name__ == '__main__':
    app.run_server(debug=True, port=8081)