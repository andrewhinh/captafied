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
import validators


# Variables
# Port for Dash app
DEFAULT_PORT = 11700

# Flask server for loading assets
server = Flask(__name__)
ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# CSS stylesheet
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

# Text + background settings
font = {
    "family": "Helvetica",
    "size": "20px",
}
colors = {
    "background": "#111111",
    "text": "#ffffff",
}

# Most number of rows of table to show when table is uploaded
max_rows = 5

# Button helper variables
button_terms = ["Incorrect", "Offensive", "Other"]
button_id_end = "-button-state"

# Examples
example_url = "https://docs.google.com/spreadsheets/d/1a6M4bOAinxuPEnFoqS6BBB-F-rgpW907B2RXebWcN78/edit#gid=1908118829"
example_request = "What does the distribution of the column 'Stars' look like?"

# Backend pipeline
pipeline = Pipeline()

# Flagging csv file
flag_csv_path = Path(__file__).parent / ".." / ".." / "flagged" / "log.csv"


# Helper functions
# Return stylized HTML text settings
def html_settings(
    fontSize=font["size"], fontWeight="normal", width="100%", height="100%", lineHeight=None, borderStyle=None
):
    temp = {
        "textAlign": "center",
        "fontFamily": font["family"],
        "color": colors["text"],
        "backgroundColor": colors["background"],
        "fontSize": fontSize,
        "fontWeight": fontWeight,
        "width": width,
        "height": height,
    }
    if lineHeight:
        temp["lineHeight"] = lineHeight
    if borderStyle:
        temp["borderStyle"] = borderStyle
    return temp


# Return stylized HTML text
def html_text(text, fontSize=font["size"], fontWeight="normal"):
    return html.Div(
        [text],
        style=html_settings(fontSize, fontWeight),
    )


# Return stylized HTML table
def html_table(table):
    return (
        html.Table(
            children=[
                html.Thead(html.Tr([html.Th(col) for col in table.columns])),
                html.Tbody(
                    [
                        html.Tr([html.Td(table.iloc[i][col]) for col in table.columns])
                        for i in range(min(len(table), max_rows))
                    ]
                ),
            ],
            style=html_settings(),
        ),
    )


# Return stylized HTML input box
def html_input(id, type, placeholder, width="100%"):
    return dcc.Input(
        id=id,
        type=type,
        placeholder=placeholder,
        debounce=True,
        style=html_settings(width=width),
    )


# Return button id
def button_id(idx):
    return button_terms[idx].lower() + button_id_end


# Convert files to pd.DataFrames
def name_to_pd(name, csv_obj=None, rest_obj=None):
    if not csv_obj and not rest_obj:
        csv_obj = rest_obj = name
    if "csv" in name:
        df = pd.read_csv(csv_obj)
    elif "tsv" in name:
        df = pd.read_csv(rest_obj, sep="\t")
    elif "xlsx" in name or "xls" in name:
        df = pd.read_excel(rest_obj)
    elif "ods" in name:
        df = pd.read_excel(rest_obj, engine="odf")
    # elif "pdf" in name:
    # df = tb.read_pdf(rest_obj, pages="all")
    # elif "html" in name:
    # df = pd.read_html(rest_obj)
    else:
        raise ValueError()
    return df


# Convert an uploaded file/typed-in URL to a pd.DataFrame
def convert_to_pd(contents=None, filename=None, url=None):
    empty_df, empty_error = None, None
    error_ending = "csv, tsv, xls(x), or ods file containing a table."
    url_error = "Please enter a valid public URL to a " + error_ending
    file_error = "Please upload a valid " + error_ending  # /pdf/html
    if contents and filename:
        try:
            _, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            df = name_to_pd(filename, io.StringIO(decoded.decode("utf-8")), io.BytesIO(decoded))
        except:
            return empty_df, html_text(file_error)
        return df, empty_error
    if url:
        if validators.url(url):
            if "https://docs.google.com/spreadsheets/" in url:  # In case this is a Google Sheets URL
                url = url.replace("/edit#gid=", "/export?format=csv&gid=")
            try:
                df = name_to_pd(url)
            except:
                return empty_df, html_text(url_error)
            return df, empty_error
        else:
            return empty_df, html_text(url_error)


# Show a table from an uploaded file/typed-in URL
def show_uploaded_table(contents=None, filename=None, url=None):
    heading = ""
    df, error = None, None
    if contents and filename:
        heading = filename
        df, error = convert_to_pd(contents, filename, None)
    if url:
        heading = url
        df, error = convert_to_pd(None, None, url)
    if df is not None and not error:
        if type(df) == pd.DataFrame:
            return html.Div([html_text(heading), html_table(df)])
        else:
            return error
    else:
        return error


# Show output
def show_output(table=None, text=None, graph=None, report=None, error=None):
    outputs = []

    if table is not None:
        outputs.extend([html_table(table)])

    if text is not None:
        outputs.extend([html_text(text)])

    if graph is not None:
        outputs.extend(
            [
                dcc.Graph(
                    figure=graph,
                ),
            ]
        )

    if report is not None:
        message, report_path = report[0], report[1]
        outputs.extend(
            [
                html_text(message),
                html.Iframe(
                    src=report_path,
                    style={"height": "1080px", "width": "100%"},
                ),
            ]
        )

    if error is not None:
        outputs.extend(
            [
                html_text(error),
            ]
        )

    return html.Div(style=html_settings(), children=outputs)


# Flag output
def flag_output(request, pred, incorrect=None, offensive=None, other=None):
    clicked = ["temp"]
    log = pd.DataFrame(
        {
            "Request": [request],
            "Output": [pred],
            "Flag": clicked,
            "Timestamp": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        }
    )
    clicked = clicked[1:]

    if incorrect:
        clicked.append("incorrect")
        log["Flag"] = clicked

    if offensive:
        clicked.append("offensive")
        log["Flag"] = clicked

    if other:
        clicked.append("other")
        log["Flag"] = clicked

    if clicked:
        log.to_csv(flag_csv_path, mode="a", index=False, header=not os.path.exists(flag_csv_path))


# Main frontend code
# Initializing dash app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=server)

# Initializing app layout
app.layout = html.Div(
    style=html_settings(),
    children=[
        # Webpage title
        html_text("Captafied", "70px", "bold"),
        # Subtitle
        html_text("Edit, query, graph, and understand your table!", "30px"),
        # Description + usage instructions
        html.Center(
            [
                dcc.Markdown(
                    [
                        """
                    Some notes and examples for the supported requests 
                    can be found [here](https://github.com/andrewhinh/captafied#usage).
                    If the output is wrong in some way, 
                    let us know by clicking the "flagging" buttons underneath.
                    We'll analyze the results and use them to improve the model! 
                    For more on how this application works, 
                    check out the [GitHub repo](https://github.com/andrewhinh/captafied) 
                    (and give it a star if you like it!).
                    """
                    ],
                    style=html_settings(width="80%"),
                )
            ]
        ),
        # Line break
        html.Br(),
        # Input: table file
        html_text("Upload your table as a csv, tsv, xls(x), or ods file:"),
        html.Center(
            [
                dcc.Upload(
                    id="before-table-file-uploaded",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style=html_settings(width="80%", height="400%", lineHeight="400%", borderStyle="dashed"),
                    multiple=True,  # Allow multiple files to be uploaded, temporary since we only want one file but necessary for repeated uses
                ),
            ]
        ),
        html.Br(),
        html.Div(id="after-table-file-uploaded"),
        html.Br(),
        # Input: table as a URL
        html.Div(
            [
                html_text("Or paste a public URL to it:"),
                html_input(id="before-table-url-uploaded", type="url", placeholder=example_url, width="80%"),
            ]
        ),
        html.Br(),
        html.Div(id="after-table-url-uploaded"),
        html.Br(),
        # Input: text request
        html.Div(
            [
                html_text("Type in a request:"),
                html_input(id="request-uploaded", type="text", placeholder=example_request, width="80%"),
            ]
        ),
        html.Br(),
        # Output: table, text, graph, report, and/or error
        html.Div(id="pred_output"),
        html.Br(),
        # Flagging buttons
        html.Button(
            button_terms[0],
            id=button_id(0),
            n_clicks=0,
            style=html_settings(width="15%"),
        ),
        html.Button(
            button_terms[1],
            id=button_id(1),
            n_clicks=0,
            style=html_settings(width="15%"),
        ),
        html.Button(
            button_terms[2],
            id=button_id(2),
            n_clicks=0,
            style=html_settings(width="15%"),
        ),
        html.Div(id="flag_output"),
        # Extra length at the bottom of the page
        html.Div([html.Br()] * 10),
    ],
)


# Event functions
# When only table file is uploaded
@app.callback(
    Output("after-table-file-uploaded", "children"),
    Input("before-table-file-uploaded", "contents"),
    State("before-table-file-uploaded", "filename"),
)
def show_uploaded_table_file(contents, filename):
    if contents and filename:
        return show_uploaded_table(contents[0], filename[0], None)


# When only table URL is uploaded
@app.callback(Output("after-table-url-uploaded", "children"), Input("before-table-url-uploaded", "value"))
def show_uploaded_table_url(url):
    if url:
        return show_uploaded_table(None, None, url)


# When both table file/url + request are uploaded
@app.callback(
    Output("pred_output", "children"),
    Input("request-uploaded", "value"),
    Input("before-table-file-uploaded", "contents"),
    State("before-table-file-uploaded", "filename"),
    Input("before-table-url-uploaded", "value"),
)
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
@app.callback(
    Output("flag_output", "children"),
    State("request-uploaded", "value"),
    State("pred_output", "children"),
    State("before-table-file-uploaded", "contents"),
    State("before-table-file-uploaded", "filename"),
    State("before-table-url-uploaded", "value"),
    Input("incorrect-button-state", "n_clicks"),
    Input("offensive-button-state", "n_clicks"),
    Input("other-button-state", "n_clicks"),
)
def flag_pred(
    request, pred, contents=None, filename=None, url=None, inc_clicked=None, off_clicked=None, oth_clicked=None
):
    if pred:
        try:
            pred = [x["props"]["children"] for x in pred["props"]["children"]]
            pred = " ".join(pred)
        except:
            pred = "Graph"

    changed_id = [p["prop_id"] for p in dash.callback_context.triggered][0]
    buttons_clicked = [
        button_id(0) in changed_id,
        button_id(1) in changed_id,
        button_id(2) in changed_id,
    ]

    df, error = None, None
    if True in buttons_clicked and contents and filename and request and pred:
        df, error = convert_to_pd(contents[0], filename[0], None)
    if True in buttons_clicked and url and request and pred:
        df, error = convert_to_pd(None, None, url)

    checks = df is not None and request is not None and pred is not None and not error
    if checks and buttons_clicked[0]:
        flag_output(request, pred, True, None, None)
    if checks and buttons_clicked[1]:
        flag_output(request, pred, None, True, None)
    if checks and buttons_clicked[2]:
        flag_output(request, pred, None, None, True)


# When report is generated and needs to be displayed
@app.server.route("/assets/<resource>")
def serve_assets(resource):
    return flask.send_from_directory(ASSETS_PATH, resource)


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


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    app.run_server(debug=True, dev_tools_hot_reload=False, port=args.port)
