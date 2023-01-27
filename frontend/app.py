# Libraries
import argparse
import base64
import datetime
import io
import os
from pathlib import Path
import shutil

from backend.inference.inference import Pipeline
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import flask
from flask import Flask
import pandas as pd
import validators


# Variables
# Global variables for downloading table and for backend
output_tables = []
requests = []
answers = []

# Port for Dash app
DEFAULT_PORT = 11700

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

# Max of table to show when table is sent to backend (From AutoViz)
max_pred_rows = 150000
max_pred_cols = 30

# Max of table to show when table is shown
max_rows = 5
max_col = 2

# Max html table heading character length (since there are usually no spaces)
max_char_length = 40 - 3  # Width of IPhone screen - "..." at end

# Example error message when multiple tables are found
multiple_files_error = "Please upload one file with only one table."

# Flask server for loading assets
server = Flask(__name__)
ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Button helper variables
flag_terms = ["incorrect", "offensive", "other"]
button_id_end = "-button-state"

# Examples
parent_path = Path(__file__).parent / ".."
table_path = parent_path / "backend" / "inference" / "tests" / "support" / "tables"
table_example = table_path / "0.csv"
url_example_path = table_path / "1.txt"
url_example = open(url_example_path).readlines()[0]
example_phrase = "use example"

# Table download settings
asset_path = parent_path / "frontend" / "assets"
download_path = asset_path / "tables"
# Make this directory if it doesn't exist
if not os.path.exists(download_path):
    os.makedirs(download_path)
zip_name = "temp"

# Flagging csv file
flag_csv_path = parent_path / "flagged" / "log.csv"

# Loading spinner
spinner = "default"

# Backend pipeline
pipeline = Pipeline()


# Helper functions
# Return stylized HTML text settings
def html_settings(
    fontSize=font["size"], fontWeight="normal", width="90%", height="100%", lineHeight=None, borderStyle=None
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


# Return stylized HTML input box
def html_input(id, type):
    return dcc.Input(
        id=id,
        type=type,
        debounce=True,
        style=html_settings(),
    )


# Return button id
def flag_button_id(idx):
    return flag_terms[idx] + button_id_end


# Show all clicked buttons
def total_clicked_buttons():
    return [p["prop_id"] for p in dash.callback_context.triggered][0]


# Convert files to pd.DataFrames
def name_to_pd(name, csv_obj=None, rest_obj=None):
    if not csv_obj and not rest_obj:  # For local files + URLs
        csv_obj = rest_obj = name
    if "csv" in name:
        df = pd.read_csv(csv_obj)
    elif "tsv" in name:
        df = pd.read_csv(rest_obj, sep="\t")
    elif "xlsx" in name or "xls" in name:
        df = pd.read_excel(rest_obj)
    elif "ods" in name:
        df = pd.read_excel(rest_obj, engine="odf")
    else:
        raise ValueError()
    return df


# Convert an uploaded file/typed-in URL to a pd.DataFrame
def convert_to_pd(contents=None, filename=None, url=None):
    empty_df, empty_error = None, None
    error_ending = "csv, tsv, xls(x), or ods file containing a table."
    url_error = "Please enter a valid public URL to a " + error_ending
    file_error = "Please upload a valid " + error_ending
    if not contents and filename:
        try:
            df = name_to_pd(filename)
        except BaseException:
            return empty_df, html_text(file_error)
        return df, empty_error
    if contents and filename:
        try:
            _, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            df = name_to_pd(filename, io.StringIO(decoded.decode("utf-8")), io.BytesIO(decoded))
        except BaseException:
            return empty_df, html_text(file_error)
        return df, empty_error
    if url:
        if validators.url(url):
            if "https://docs.google.com/spreadsheets/" in url:  # In case this is a Google Sheets URL
                url = url.replace("/edit#gid=", "/export?format=csv&gid=")
            try:
                df = name_to_pd(url)
            except BaseException:
                return empty_df, html_text(url_error)
            return df, empty_error
        else:
            return empty_df, html_text(url_error)


# Set maximum number of rows/columns to work with for a table
def set_max_rows_cols(df):
    if len(df) > max_pred_rows:
        df = df.iloc[:max_pred_rows]
    if len(df.columns) > max_pred_cols:
        df = df.iloc[:, :max_pred_cols]
    return df


# Show a table from an uploaded file/typed-in URL
def show_table(contents=None, filename=None, url=None, table_answer=False):
    heading = ""
    df, error = None, None
    if not contents and filename:
        heading = str(filename).split("/")[-1]
        df, error = convert_to_pd(None, str(filename), None)
    if contents and filename:
        heading = filename
        df, error = convert_to_pd(contents, filename, None)
    if url:
        heading = url
        df, error = convert_to_pd(None, None, url)
    if table_answer is not None:
        df = table_answer
    if df is not None and not error:
        items = []
        if heading:
            if validators.url(heading):
                items.append(html.A(html_text("URL"), href=heading, target="_blank"))
            else:
                if len(heading) > max_char_length:
                    heading = heading[:max_char_length] + "..."
                items.append(html_text(heading))
        rows_to_show = min(len(df), max_rows)
        cols_to_show = min(len(df.columns), max_col)
        items.append(
            html.Table(
                children=[
                    html.Thead(html.Tr([html.Th(df.columns[col]) for col in range(cols_to_show)])),
                    html.Tbody(
                        [
                            html.Tr([html.Td(df.iloc[i][df.columns[col]]) for col in range(cols_to_show)])
                            for i in range(min(len(df), rows_to_show))
                        ]
                    ),
                ],
                style=html_settings(),
            )
        )
        return html.Div(items)
    else:
        return error


# Manage/show output
def manage_output(code=None, tables=None, texts=None, graphs=None, images=None, report=None):
    outputs = []

    if code:
        global answers
        answers.append(code)

    if tables:
        elements = [
            dcc.Download(id="download-table-csv"),
            html.Button(
                "download" if len(tables) == 1 else "download all",
                id="download-table" + button_id_end,
                n_clicks=0,
                style=html_settings(width="50%"),
            ),
        ]
        global output_tables
        for table in tables:
            output_tables.append(table)
            elements.append(show_table(table_answer=table))
            elements.append(html.Br())
        elements = elements[:-1]
        outputs.extend(
            [
                html.Center(elements),
            ]
        )

    if texts:
        elements = []
        for text in texts:
            elements.append(html_text(text))
            elements.append(html.Br())
        elements = elements[:-1]
        outputs.extend(
            [
                html.Center(elements),
            ]
        )

    if graphs:
        elements = []
        for graph in graphs:
            elements.append(
                dcc.Graph(
                    figure=graph,
                    style=html_settings(),
                ),
            )
            elements.append(html.Br())
        elements = elements[:-1]
        outputs.extend(
            [
                html.Center(elements),
            ]
        )

    if images:
        elements = []
        for image in images:
            elements.append(
                html.Img(
                    src=image,
                    style=html_settings(),
                ),
            )
            elements.append(html.Br())
        outputs.extend(
            [
                html.Center(elements),
            ]
        )

    if report:
        message, report_path = report[0], report[1]
        outputs.extend(
            [
                html.Center(
                    [
                        html_text(message),
                        html.Iframe(
                            src=report_path,
                            style=html_settings(height="540px"),
                        ),
                        html.Br(),
                    ]
                ),
            ]
        )

    return html.Div(style=html_settings(width="100%"), children=outputs)


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

    return (html_text("Done! Thanks for your feedback."),)


# Main frontend code
# Initializing dash app
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    server=server,
    suppress_callback_exceptions=True,
)

# Initializing app layout
app.layout = html.Div(
    style=html_settings(width="100%"),
    children=[
        # Webpage title
        html.Center(
            [
                html_text("Captafied", "70px", "bold"),
                # Subtitle
                html_text("Edit, query, graph, and understand your table!", "30px"),
                # Description + usage instructions
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
                    style=html_settings(),
                ),
                # Line break
                html.Br(),
                # Input: table file
                html_text("Upload your table as a csv, tsv, xls(x), or ods file:"),
                html.Button(
                    example_phrase,
                    id="file-upload-example",
                    n_clicks=0,
                    style=html_settings(width="50%"),
                ),
                html.Br(),
                html.Br(),
                dcc.Upload(
                    id="before-table-file-uploaded",
                    children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                    style=html_settings(height="400%", lineHeight="400%", borderStyle="dashed"),
                    # multiple=True,  # Allow multiple files to be uploaded, temporary since we only want one file but necessary for repeated uses
                ),
                html.Br(),
                dcc.Loading(
                    type=spinner,
                    children=html.Div(id="after-table-file-uploaded"),
                ),
                html.Br(),
                # Input: table as a URL
                html_text("Or paste a public URL to it:"),
                html.Button(
                    example_phrase,
                    id="url-upload-example",
                    n_clicks=0,
                    style=html_settings(width="50%"),
                ),
                html.Br(),
                html.Br(),
                html_input(id="before-table-url-uploaded", type="url"),
                html.Br(),
                html.Br(),
                dcc.Loading(
                    type=spinner,
                    children=html.Div(id="after-table-url-uploaded"),
                ),
                html.Br(),
                # Input: text request
                html_text("Type in a request:"),
                # Width = 390px for Typing SVG for iphone devices
                dcc.Markdown(
                    [
                        """
                [![Typing SVG](https://readme-typing-svg.demolab.com?font=Helvetica&duration=2500&pause=1000&center=true&vCenter=true&width=390&lines=Add+a+column+that+averages+forks+and+stars.;Which+rows+have+more+than+1000+stars%3F;Does+Transformers+have+the+most+stars%3F;What+does+the+distribution+of+stars+look+like%3F;What+does+the+Transformers+icon+look+like%3F;Show+the+summary+clusters.)](https://git.io/typing-svg)
                """
                    ],
                    style=html_settings(),
                ),
                html_input(id="request-uploaded", type="text"),
                html.Br(),
                html.Br(),
                # Output: table, text, graph, report, and/or error
                dcc.Loading(
                    type=spinner,
                    children=html.Div(id="pred_output"),
                ),
                html.Br(),
                # Flagging buttons
                html.Button(
                    flag_terms[0],
                    id=flag_button_id(0),
                    n_clicks=0,
                    style=html_settings(width="50%"),
                ),
                html.Br(),
                html.Button(
                    flag_terms[1],
                    id=flag_button_id(1),
                    n_clicks=0,
                    style=html_settings(width="50%"),
                ),
                html.Br(),
                html.Button(
                    flag_terms[2],
                    id=flag_button_id(2),
                    n_clicks=0,
                    style=html_settings(width="50%"),
                ),
                html.Br(),
                html.Br(),
                dcc.Loading(
                    type=spinner,
                    children=html.Div(id="flag_output"),
                ),
                # Extra length at the bottom of the page
                html.Div([html.Br()] * 10),
            ]
        ),
    ],
)


# Event functions
# When only table file is uploaded
@app.callback(
    Output("after-table-file-uploaded", "children"),
    Input("before-table-file-uploaded", "contents"),
    State("before-table-file-uploaded", "filename"),
    Input("file-upload-example", "n_clicks"),
)
def show_uploaded_table_file(contents, filename, example_clicked):
    if "file-upload-example" in total_clicked_buttons():
        return show_table(None, table_example, None, None)
    if contents and filename:
        if type(contents) == list and type(filename) == list:
            if len(contents) > 1 and len(filename) > 1:
                return html_text(multiple_files_error)
        return show_table(contents, filename, None, None)


# Show example/uploaded table URL
@app.callback(
    Output("after-table-url-uploaded", "children"),
    Input("before-table-url-uploaded", "value"),
    Input("url-upload-example", "n_clicks"),
)
def show_uploaded_table_url(url, example_clicked):
    if "url-upload-example" in total_clicked_buttons():
        return show_table(None, None, url_example, None)
    if url:
        return show_table(None, None, url, None)


# When both table file/url + request are uploaded
@app.callback(
    Output("pred_output", "children"),
    State("after-table-file-uploaded", "children"),
    State("after-table-url-uploaded", "children"),
    Input("request-uploaded", "value"),
    Input("before-table-file-uploaded", "contents"),
    State("before-table-file-uploaded", "filename"),
    Input("before-table-url-uploaded", "value"),
)
def get_prediction(show_file, show_url, request, contents, filename, url):
    df, error = None, None
    checks = [
        contents and filename and request,
        url and request,
        show_file and request,
        show_url and request,
    ]
    if checks[0]:
        df, error = convert_to_pd(contents[0], filename[0], None)
    if checks[1]:
        df, error = convert_to_pd(None, None, url)
    if (not checks[0] and not checks[1]) and checks[2]:
        df, error = convert_to_pd(None, str(table_example), None)
    if (not checks[0] and not checks[1]) and checks[3]:
        df, error = convert_to_pd(None, None, str(url_example))
    if df is not None and not error:
        global output_tables
        output_tables = []
        df = set_max_rows_cols(df)
        global requests
        requests.append(request)
        code, tables, texts, graphs, images, report = pipeline.predict(df, requests, answers)
        return manage_output(code, tables, texts, graphs, images, report)


# When download button is clicked
@app.callback(
    Output("download-table-csv", "data"),
    Input("download-table" + button_id_end, "n_clicks"),
    prevent_initial_call=True,
)
def download_table(n_clicks):
    if "download-table" + button_id_end in total_clicked_buttons():
        if len(output_tables) > 0 and len(output_tables) < 2:
            return dcc.send_data_frame(output_tables[0].to_csv, "table.csv")
        elif len(output_tables) > 1:
            for idx, output_table in enumerate(output_tables):
                output_table.to_csv(str(download_path / str("table_" + str(idx) + ".csv")), index=False)
            zip_file_name = str(asset_path / zip_name)
            shutil.make_archive(zip_file_name, "zip", str(download_path))
            return dcc.send_file(zip_file_name + ".zip")
        else:
            pass


# When flagging button(s) is/are clicked
@app.callback(
    Output("flag_output", "children"),
    State("after-table-file-uploaded", "children"),
    State("after-table-url-uploaded", "children"),
    State("request-uploaded", "value"),
    State("pred_output", "children"),
    State("before-table-file-uploaded", "contents"),
    State("before-table-file-uploaded", "filename"),
    State("before-table-url-uploaded", "value"),
    Input("incorrect-button-state", "n_clicks"),
    Input("offensive-button-state", "n_clicks"),
    Input("other-button-state", "n_clicks"),
)
def flag_pred(show_file, show_url, request, pred, contents, filename, url, inc_clicked, off_clicked, oth_clicked):
    if pred:
        try:
            pred = [x["props"]["children"] for x in pred["props"]["children"]]
            pred = " ".join(pred)
        except BaseException:
            pred = "Graph"

    changed_id = total_clicked_buttons()
    buttons_clicked = [
        flag_button_id(0) in changed_id,
        flag_button_id(1) in changed_id,
        flag_button_id(2) in changed_id,
    ]

    checks = [
        True in buttons_clicked and pred,
        contents and filename and request,
        url and request,
        show_file and request,
        show_url and request,
    ]

    df, error = None, None
    if checks[0] and checks[1]:
        df, error = convert_to_pd(contents[0], filename[0], None)
    if checks[0] and checks[2]:
        df, error = convert_to_pd(None, None, url)
    if checks[0] and (not checks[1] and not checks[2]) and checks[3]:
        df, error = convert_to_pd(None, str(table_example), None)
    if checks[0] and (not checks[1] and not checks[2]) and checks[4]:
        df, error = convert_to_pd(None, None, str(url_example))

    check = (show_file or show_url) and df is not None and request and pred is not None and not error
    if check and buttons_clicked[0]:
        return flag_output(request, pred, True, None, None)
    if check and buttons_clicked[1]:
        return flag_output(request, pred, None, True, None)
    if check and buttons_clicked[2]:
        return flag_output(request, pred, None, None, True)


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
