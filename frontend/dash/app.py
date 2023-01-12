# Libraries
import argparse
import base64
import datetime
import io
import os

from backend.inference.inference import Pipeline
import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import flask
from flask import Flask
import os
import pandas as pd
from pathlib import Path
import tabula as tb


# Variables
# Port for Dash app
DEFAULT_PORT = 11700

# Flask server for loading assets
server = Flask(__name__)
ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')

# CSS stylesheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Text + background colors
colors = {
    'background': '#111111',
    'text': '#ffffff',
}

# Examples
example_url = "https://docs.google.com/spreadsheets/d/1a6M4bOAinxuPEnFoqS6BBB-F-rgpW907B2RXebWcN78/edit#gid=1908118829"
example_request = "What does the distribution of the column 'Stars' look like?"

# Most number of rows of table to show when table is uploaded
max_rows = 5

# Backend pipeline
pipeline = Pipeline()

# Flagging csv file
flag_csv_path = Path(__file__).parent / '..' / '..' / 'flagged' / 'log.csv'


# Main frontend code
# Initializing dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

# Initializing app layout
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    # Webpage title
    html.H1('Captafied',
            style={
                'textAlign': 'center',
                'fontFamily': 'Helvetica',
                'fontSize': '70px',
                'fontWeight': 'bold',
                'color': colors['text'],
            }
    ),

    # Subtitle
    html.H2('Edit, query, graph, and understand your table!',
            style={
                'textAlign': 'center',
                'fontFamily': 'Helvetica',
                'fontSize': '30px',
                'color': colors['text'],
            }
    ),

    # Line break
    html.Br(), 
    
    # Input: table file
    html.H3('Upload your table as a file:',
            style={
                'textAlign': 'center',
                'fontFamily': 'Helvetica',
                'fontSize': '20px',
                'color': colors['text'],
            }
    ),
    html.Center([
        dcc.Upload(id='before-table-file-uploaded',
                   children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                   ]),
                   style={
                        'textAlign': 'center',
                        'fontFamily': 'Helvetica',
                        'fontSize': '20px',
                        'width': '40%',
                        'height': '100px',
                        'lineHeight': '100px',
                        'borderStyle': 'dashed',
                        'color': colors['text'],
                   },
                   multiple=True, # Allow multiple files to be uploaded, temporary since we only want one file but necessary for repeaeted uses
        ),
    ]),
    html.Br(),
    html.Div(id='after-table-file-uploaded'),

    html.Br(),

    # Input: table as a URL
    html.Div([
        html.H3('Or paste a public Google Sheets URL:',
            style={
                'textAlign': 'center',
                'fontFamily': 'Helvetica',
                'fontSize': '20px',
                'color': colors['text'],
            }
        ),
        dcc.Input(id='before-table-url-uploaded',
                  type='url',
                  placeholder=example_url,
                  debounce=True,
                  style={
                    'textAlign': 'center',
                    'fontSize': '20px',
                    'color': colors['text'],
                    'backgroundColor': colors['background']
                  }),
        ],
        style={
            'textAlign': 'center',
            'fontSize': '20px',
            'color': colors['text'],
        }
    ),
    html.Br(),
    html.Div(id='after-table-url-uploaded'),

    html.Br(),

    # Input: text request
    html.Div([
        html.H3('Type in a request:',
            style={
                'textAlign': 'center',
                'fontFamily': 'Helvetica',
                'fontSize': '20px',
                'color': colors['text'],
            }
        ),
        dcc.Input(id='request-uploaded',
                  type='text',
                  placeholder=example_request,
                  debounce=True,
                  style={
                    'textAlign': 'center',
                    'fontSize': '20px',
                    'color': colors['text'],
                    'backgroundColor': colors['background']
                  }),
        ],
        style={
            'textAlign': 'center',
            'fontSize': '20px',
            'color': colors['text'],
        }
    ),

    html.Br(),

    # Output: table, text, graph, report, and/or error
    html.Div(id='pred_output'),

    html.Br(),

    # Flagging buttons
    html.Div([
        html.Button(id='incorrect-button-state', 
                    n_clicks=0,
                    children='Incorrect',
                    style={'color': colors['text']}), # Incorrect flag button
        html.Button(id='offensive-button-state', 
                    n_clicks=0, 
                    children='Offensive',
                    style={'color': colors['text']}), # Offensive flag button
        html.Button(id='other-button-state', 
                    n_clicks=0, 
                    children='Other',
                    style={'color': colors['text']}), # Other flag button
        ],
        style={
            'textAlign': 'center',
            'fontSize': '20px',
            'color': colors['text'],
        }
    ),
    html.Div(id='flag_output'),

    # Extra length at the bottom of the page
    html.Div([
        html.Br(),
    ]*10),
])


# Helper functions
# Convert an uploaded file/typed-in URL to a pd.DataFrame 
def convert_to_pd(contents=None, filename=None, url=None):
    empty_df, empty_error = None, None
    if contents and filename:
        _, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'tsv' in filename:
                df = pd.read_csv(io.BytesIO(decoded), sep='\t')
            elif 'xlsx' in filename or 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
            elif "ods" in filename:
                df = pd.read_excel(io.BytesIO(decoded), engine="odf")
            elif "pdf" in filename:
                df = tb.read_pdf(io.BytesIO(decoded), pages='all')
            elif "html" in filename:
                df = pd.read_html(io.BytesIO(decoded))
            else:
                raise ValueError("File type not supported.")
        except:
            return empty_df, html.Div(style={'backgroundColor': colors['background']}, children=[
                'There was an error processing this file. Please submit a csv/tsv/xls(x)/ods/pdf/html file.'
            ])
        return df, empty_error
    if url:
        url = url.replace('/edit#gid=', '/export?format=csv&gid=') # In case this is a url
        try:
            df = pd.read_csv(url)
        except:
            return empty_df, html.Div(style={'backgroundColor': colors['background']}, children=[
                'There was an error processing this url. Please submit a valid public Google Sheets URL.'
            ])
        return df, empty_error
    return empty_df, html.Div(style={'backgroundColor': colors['background']}, children=[
        'Please submit a valid public Google Sheets URL or csv/tsv/xls(x)/ods/pdf/html file.'
    ])

# Show a table from an uploaded file/typed-in URL
def show_uploaded_table(contents=None, filename=None, url=None):
    heading = ''
    df, error = None, None
    if filename:
        heading = filename
        df, error = convert_to_pd(contents, filename, None)
    if url:
        heading = url
        df, error = convert_to_pd(None, None, url)
    try:
        if type(df) == pd.DataFrame:
            return html.Div(style={'backgroundColor': colors['background']}, children=[
                html.H4(heading,
                        style={
                            'textAlign': 'center',
                            'fontFamily': 'Helvetica',
                            'font-size': '20px',
                            'color': colors['text'],
                        }),
                html.Table(children=[
                    html.Thead(
                        html.Tr([html.Th(col) for col in df.columns])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(df.iloc[i][col]) for col in df.columns
                        ]) for i in range(min(len(df), max_rows))
                    ])
                    ],
                    style={
                        'color': colors['text'],
                        'backgroundColor': colors['background'],
                    }
                )
            ])
    except:
        return error
    
# Show output
def show_output(table=None, text=None, graph=None, report=None, error=None):
    outputs = []
    
    if table is not None:
        outputs.extend([html.Table(children=[
                        html.Thead(
                            html.Tr([html.Th(col) for col in table.columns])
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td(table.iloc[i][col]) for col in table.columns
                            ]) for i in range(min(len(table), max_rows))
                        ])
                        ],
                        style={
                            'color': colors['text'],
                            'backgroundColor': colors['background'],
                            'height': 'auto',
                            'width': 'auto',
                        }
                    )])
    
    if text is not None:
        outputs.extend([
            html.H5(text,
                    style={
                        'textAlign': 'center',
                        'fontFamily': 'Helvetica',
                        'font-size': '20px',
                        'color': colors['text'],
                    }),
        ])
    
    if graph is not None:
        outputs.extend([
            html.Img(graph)
        ])
    
    if report is not None:
        message, report_path = report[0], report[1]
        outputs.extend([html.H5(message,
                                style={
                                    'textAlign': 'center',
                                    'fontFamily': 'Helvetica',
                                    'font-size': '20px',
                                    'color': colors['text'],
                                }
                        ),
                        html.Iframe(
                                src=report_path,
                                style={"height": "1080px", "width": "100%"},
                        )]
        )
    
    if error is not None:
        outputs.extend([
            html.H5(error,
                    style={
                        'textAlign': 'center',
                        'font-size': '20px',
                        'color': colors['text'],
                    }),
        ])

    return html.Div(style={'backgroundColor': colors['background']}, children=outputs)

# Flag output
def flag_output(request, pred, incorrect=None, offensive=None, other=None):
    clicked = ["temp"]
    log = pd.DataFrame({'Request': [request], 
                        'Output': [pred], 
                        'Flag': clicked, 
                        'Timestamp': [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
                    })
    clicked = clicked[1:]

    if incorrect:
        clicked.append('incorrect')
        log['Flag'] = clicked
    
    if offensive:
        clicked.append('offensive')
        log['Flag'] = clicked

    if other:
        clicked.append('other')
        log['Flag'] = clicked

    if clicked:
        log.to_csv(flag_csv_path, mode='a', index=False, header=not os.path.exists(flag_csv_path))


# Event functions
# When only table file is uploaded
@app.callback(Output('after-table-file-uploaded', 'children'),
              Input('before-table-file-uploaded', 'contents'),
              State('before-table-file-uploaded', 'filename'))
def show_uploaded_table_file(contents, filename):
    if contents and filename:
        return show_uploaded_table(contents[0], filename[0], None)

# When only table URL is uploaded
@app.callback(Output('after-table-url-uploaded', 'children'),
              Input('before-table-url-uploaded', 'value'))
def show_uploaded_table_url(url):
    if url:
        return show_uploaded_table(None, None, url)

# When both table file/url + request are uploaded
@app.callback(Output('pred_output', 'children'),
              Input('request-uploaded', 'value'),
              Input('before-table-file-uploaded', 'contents'),
              State('before-table-file-uploaded', 'filename'),
              Input('before-table-url-uploaded', 'value'))
def get_prediction(request, contents=None, filename=None, url=None):
    df, error = None, None
    if contents and filename and request:
        df, error = convert_to_pd(contents[0], filename[0], None)
    if df is None:
        if url and request:
            df, error = convert_to_pd(None, None, url)
    if df is not None and not error:
        table, text, graph, report, pred_error = pipeline.predict(df, request)
        return show_output(table, text, graph, report, pred_error)

# When flagging button(s) is/are clicked
@app.callback(Output('flag_output', 'children'),
              State('request-uploaded', 'value'),
              State('pred_output', 'children'),
              State('before-table-file-uploaded', 'contents'),
              State('before-table-file-uploaded', 'filename'),
              State('before-table-url-uploaded', 'value'),
              Input('incorrect-button-state', 'n_clicks'),
              Input('offensive-button-state', 'n_clicks'),
              Input('other-button-state', 'n_clicks'))
def flag_pred(request, pred, contents=None, filename=None, url=None, inc_clicked=None, off_clicked=None, oth_clicked=None):
    if pred:
        pred = [x['props']['children'] if x['props']['children'] else x['props']['src'] for x in pred['props']['children']]
        pred = " ".join(pred)

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    button_clicked = ['incorrect-button-state' in changed_id, 
                      'offensive-button-state' in changed_id,
                      'other-button-state' in changed_id,
    ]
    
    df, error = None, None   
    if True in button_clicked and contents and filename and request and pred:
        df, error = convert_to_pd(contents[0], filename[0], None)
    if True in button_clicked and url and request and pred:
        df, error = convert_to_pd(None, None, url)

    checks = df is not None and request is not None and pred is not None and not error
    if checks and button_clicked[0]:
        flag_output(request, pred, True, None, None)
    if checks and button_clicked[1]:
        flag_output(request, pred, None, True, None)
    if checks and button_clicked[2]:
        flag_output(request, pred, None, None, True)

# When report is generated and needs to be displayed
@app.server.route('/assets/<resource>')
def serve_assets(resource):
    return flask.send_from_directory(ASSETS_PATH, resource)

"""
# For debugging, display the raw contents provided by the web browser
html.Div('Raw Content'),
html.Pre(contents[0:200] + '...', style={
    'whiteSpace': 'pre-wrap',
    'wordBreak': 'break-all'
})
"""


# Code execution
def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )

    return parser

if __name__ == '__main__':
    parser = _make_parser()
    args = parser.parse_args()
    app.run_server(debug=True, dev_tools_hot_reload=False, port=args.port)