import base64
import io

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html, dash_table

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(children='Captafied',
            style={
                'text-align': 'center',
                'font-family': 'Helvetica',
                'font-size': '70px',
                'margin-top': '10px',
                'font-weight': 'bold',
            }),

    html.Div(children='''Edit or ask any question about your table!''',
             style={
                 'font-family': 'Helvetica',
                 'font-size': '20px',
                 'margin-left' : '90px',
                 'padding-left' : '90px'
             }),

    dcc.Upload(
        id='datatable-upload',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '40%',
            'height': '150px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'padding': '60px',
            'line-height': '150px',
            'font-size': '20px'
        },
    ),

    dcc.Input(id='input-on-submit',
              type='text',
              placeholder= 'Request Captafied to modify data.',
              style={
                  'width': '40%',
                  'margin-top': '10px',
                  'margin-right': '10px',
                  'margin-bottom': '20px'
              }
    ),

    html.Nobr(),

    html.Button('Submit',
            id='request-button',
            n_clicks=0,
            style={'margin-top': '10px',
                    'color': 'green',
                    'border-color': 'green',
            }
    ),

    dash_table.DataTable(id='datatable-upload-container'),
    dcc.Graph(id='datatable-upload-graph',
              style={
                  'position': 'absolute',
                  'top': '10%',
                  'left': '50%',
                  'height': '50%',
                  'width': '50%'
              })
])


def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        # Assume that the user uploaded a CSV file
        return pd.read_csv(
            io.StringIO(decoded.decode('utf-8')))
    elif 'xls' in filename:
        # Assume that the user uploaded an excel file
        return pd.read_excel(io.BytesIO(decoded))


@app.callback(Output('datatable-upload-container', 'data'),
              Output('datatable-upload-container', 'columns'),
              Input('datatable-upload', 'contents'),
              State('datatable-upload', 'filename'))
def update_output(contents, filename):
    if contents is None:
        return [{}], []
    df = parse_contents(contents, filename)
    return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]


@app.callback(Output('datatable-upload-graph', 'figure'),
              Input('datatable-upload-container', 'data'))
def display_graph(rows):
    df = pd.DataFrame(rows)
    if (df.empty or len(df.columns) < 1):
        return {
            'data': [{
                'x': [],
                'y': [],
                'type': 'bar'
            }]
        }
    return {
        'data': [{
            'x': df[df.columns[0]],
            'y': df[df.columns[1]],
            'type': 'bar'
        }]
    }

if __name__ == '__main__':
    app.run_server(debug=True)