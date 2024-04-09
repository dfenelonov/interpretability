import dash_ag_grid as dag
from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
import dash_bootstrap_components as dbc

app = Dash(__name__)

df = pd.read_excel('df_thres.xlsx')

app.layout = html.Div(
    [dbc.Button('Submit', size='sm', id='btn'),
        dcc.Markdown("Example of using `rowData` in a callback with an editable grid"),
        dag.AgGrid(
            id="editing-grid2",
            columnDefs=[{ 'field': 'Feature', 'editable': False},
                        { 'field': 'Threshold', 'editable': False},
                        { 'field': 'new_Threshold', 'editable': True}],
            rowData=df.to_dict("records"),
            columnSize="sizeToFit",
        ),
        html.Div(id="editing-grid-output2"),
    ],
    style={"margin": 20},
)


@app.callback(
    Output("editing-grid2", "rowData"),
    Input("btn", "n_clicks"),
    State("editing-grid2", "cellValueChanged"),
    State("editing-grid2", "rowData"))

def update(n_clicks, cel_Value, rows):
    dff = pd.DataFrame(rows)
    dff['new_Threshold'] = pd.to_numeric(dff['new_Threshold'])
    row = pd.to_numeric(cel_Value["rowId"])
    newValue = pd.to_numeric(cel_Value["newValue"])
    dff['new_Threshold'].loc[row] = newValue*10
    
    if n_clicks:
        return dff.to_dict("records")
    else: 
        return no_update

if __name__ == "__main__":
    app.run_server(debug=False, port=8081)