# Libraries
import base64
import io
import os

import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import pandas as pd
from pathlib import Path
import tabula as tb

from backend.inference.inference import Pipeline


# Variables
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
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Initializing app layout
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    # Webpage title
    html.H1('Captafied',
            style={
                'textAlign': 'center',
                'fontFamily': 'Helvetica',
                'fontSize': '70px',
                'margin-top': '10px',
                'fontWeight': 'bold',
                'color': colors['text'],
            }),

    # Subtitle
    html.Div('Edit, query, graph, and understand your table!',
             style={
                'textAlign': 'center',
                'fontFamily': 'Helvetica',
                'fontSize': '20px',
                #'margin-left' : '90px',
                #'padding-left' : '90px',
                'color': colors['text'],                 
            }),

    # Line break
    html.Br(), 

    # Input: table file
    dcc.Upload(id='before-table-file-uploaded',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            #'width': '40%',
            'height': '150px',
            'lineHeight': '150px',
            'borderWidth': '1px',
            'borderRadius': '5px',
            'padding': '60px',
            'textAlign': 'center',
            'fontSize': '20px',
            'borderStyle': 'dashed',
            'color': colors['text'],
        },
        multiple=True, # Allow multiple files to be uploaded
    ),
    html.Div(id='after-table-file-uploaded'),

    html.Br(),

    # Input: table as a URL
    html.Label('Or paste a public Google Sheets URL:',
               style={
                'textAlign': 'center',
                'fontSize': '20px',
                'color': colors['text'],
               }),
    dcc.Input(id='before-table-url-uploaded',
              type='text',
              placeholder=example_url,
              style={
                'textAlign': 'center',
                'fontSize': '20px',
                #'width': '40%',
                'margin-top': '10px',
                'margin-right': '10px',
                'margin-bottom': '20px',
                'color': colors['text'],
                'backgroundColor': colors['background'],
              }),
    html.Button(id='process-url-button-state', n_clicks=0, children='Process URL'),
    html.Div(id='after-table-url-uploaded'),
    
    # Input: text request
    html.Label('Type your request here:',
               style={
                'textAlign': 'center',
                'fontSize': '20px',
                
                'color': colors['text'],
               }),
    dcc.Input(id='before-request-uploaded', 
              type='text',
              placeholder=example_request,
              style={
                'textAlign': 'center',
                'fontSize': '20px',
                #'width': '40%',
                'margin-top': '10px',
                'margin-right': '10px',
                'margin-bottom': '20px',
                'color': colors['text'],
                'backgroundColor': colors['background'],
              }),
    html.Button(id='ask-button-state', 
                n_clicks=0, 
                style={'margin-top': '10px',
                       'color': 'green',
                       'border-color': 'green',
                },
                children='Ask'),

    # Output: table, text, graph, report, and/or error
    html.Div(id='pred_output'),

    # Flagging buttons
    html.Button(id='incorrect-button-state', n_clicks=0, children='Incorrect'), # Incorrect flag button
    html.Button(id='offensive-button-state', n_clicks=0, children='Offensive'), # Offensive flag button
    html.Button(id='other-button-state', n_clicks=0, children='Other'), # Other flag button
    html.Div(id='flag_output'),
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
                html.H5(heading,
                        style={
                            'textAlign': 'center',
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
                        'height': 'auto',
                        'width': 'auto',
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
                        'font-size': '20px',
                        'color': colors['text'],
                    }),
        ])
    
    if graph is not None:
        outputs.extend([
            html.Img(graph)
        ])
    
    if report is not None:
        message = report[0]
        report = report[1]
        outputs.extend([html.H5(message,
                        style={
                            'textAlign': 'center',
                            'font-size': '20px',
                            'color': colors['text'],
                        }),
                        html.Iframe(
                            src=report,  # must be under assets/ to be properly served
                            style={"height": "1080px", "width": "100%"},
                        )])
    
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
    log = pd.DataFrame({'request': [request], 'prediction': [pred]})
    
    if incorrect:
        add = pd.DataFrame({'incorrect': True})
        log['incorrect'] = add
    
    if offensive:
        add = pd.DataFrame({'offensive': True})
        log['offensive'] = add

    if other:
        add = pd.DataFrame({'other': True})
        log['other'] = add

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
              Input('process-url-button-state', 'n_clicks'),
              State('before-table-url-uploaded', 'value'))
def show_uploaded_table_url(clicked, url):
    if clicked and url:
        return show_uploaded_table(None, None, url)

# When both table file/url + request are uploaded
@app.callback(Output('pred_output', 'children'),
              Input('ask-button-state', 'n_clicks'),
              State('before-request-uploaded', 'value'),
              Input('before-table-file-uploaded', 'contents'),
              State('before-table-file-uploaded', 'filename'),
              State('before-table-url-uploaded', 'value'))
def get_prediction(clicked, request, contents=None, filename=None, url=None):
    df, error = None, None
    if clicked and contents and filename and request:
        df, error = convert_to_pd(contents[0], filename[0], None)
    if clicked and url and request:
        df, error = convert_to_pd(None, None, url)
    if df is not None and not error:
        table, text, graph, report, pred_error = pipeline.predict(df, request)
        return show_output(table, text, graph, report, pred_error)

# When flagging button(s) is/are clicked
@app.callback(Output('flag_output', 'children'),
              State('before-request-uploaded', 'value'),
              State('pred_output', 'children'),
              State('before-table-file-uploaded', 'contents'),
              State('before-table-file-uploaded', 'filename'),
              State('before-table-url-uploaded', 'value'),
              Input('incorrect-button-state', 'n_clicks'),
              Input('offensive-button-state', 'n_clicks'),
              Input('other-button-state', 'n_clicks'))
def flag_pred(request, pred, contents=None, filename=None, url=None, inc_clicked=None, off_clicked=None, oth_clicked=None):
    df, error = None, None
    button_clicked = (inc_clicked or off_clicked or oth_clicked)
    if button_clicked and contents and filename and request:
        df, error = convert_to_pd(contents[0], filename[0], None)
    if button_clicked and url and request:
        df, error = convert_to_pd(None, None, url)

    checks = df is not None and not error
    if checks and inc_clicked:
        flag_output(request, pred, True, None, None)
    if checks and off_clicked:
        flag_output(request, pred, None, True, None)
    if checks and oth_clicked:
        flag_output(request, pred, None, None, True)

"""
# For debugging, display the raw contents provided by the web browser
html.Div('Raw Content'),
html.Pre(contents[0:200] + '...', style={
    'whiteSpace': 'pre-wrap',
    'wordBreak': 'break-all'
})
"""


# Code execution
if __name__ == '__main__':
    app.run_server(debug=True, port=11701)
if __name__ == '__main__':
    app.run_server(debug=True)