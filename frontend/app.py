# Libraries
import base64
import datetime
import io
import json
import logging
import os
from pathlib import Path
import shutil

import boto3
from botocore import UNSIGNED
from botocore.config import Config
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dotenv import load_dotenv
import flask
from flask import Flask
import pandas as pd
import plotly.io as pio
import requests as req
from util import encode_b64_image, open_image
import validators
import waitress

# Server variables
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")
DEFAULT_PORT = 11700  # Port for Dash app

output_tables, requests, answers = [], [], []  # Global variables for downloading table and for backend
max_pred_rows, max_pred_cols = 150000, 30  # Max of table to show when table is sent to backend (From AutoViz)
multiple_files_error = (
    "Please upload one file with only one table."  # Example error message when multiple tables are found
)

asset_server = Flask(__name__)  # Flask server for loading assets
ASSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Style variables
external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]  # CSS stylesheet
font = {
    "family": "Helvetica",
    "size": "16px",
}  # Font family and size
colors = {
    "background": "#28282B",
    "text": "#FBFAF5",
}  # Background and text colors

max_rows, max_col = 3, 3  # Max of table to show when table is shown on frontend
max_char_length = (
    40 - 3
)  # Max html table heading character length (since there are usually no spaces): Width of IPhone screen - "..." at end
flag_terms = ["incorrect", "offensive", "other"]  # Flagging terms
button_id_end = "-button-state"  # Button id ending

# Examples
parent_path = Path(__file__).parent / ".."
examples_path = parent_path / "examples"
table_path = examples_path / "tables"
table_file_example = table_path / "0.csv"
table_url_example = table_path / "1.txt"
table_url_example = open(table_url_example).readlines()[0]
image_path = examples_path / "images"
image_file_example = image_path / "0.png"
image_url_example = image_path / "8.txt"
image_url_example = open(image_url_example).readlines()[0]
example_phrase = "use example"

# Table download settings
asset_path = parent_path / "frontend" / "assets"
download_path = asset_path / "tables"
if not os.path.exists(download_path):  # Make this directory if it doesn't exist
    os.makedirs(download_path)
zip_name = "temp"

# S3 Setup
write_bucket = "captafied-ydata-report"
report_name = "report.html"
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

# Flagging csv file
flag_csv_path = parent_path / "flagged" / "log.csv"

# Loading spinner
spinner = "default"

# Logger
logging.basicConfig(level=logging.INFO)


# Helper functions
def html_settings(
    fontSize=font["size"], fontWeight="normal", width="90%", height="100%", lineHeight=None, borderStyle=None
):  # Return stylized HTML text settings
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


def html_text(text, fontSize=font["size"], fontWeight="normal"):  # Return stylized HTML text
    return html.Div(
        [text],
        style=html_settings(fontSize, fontWeight),
    )


def html_input(id, type, placeholder):  # Return stylized HTML input box
    return dcc.Input(
        id=id,
        type=type,
        placeholder=placeholder,
        debounce=True,
        style=html_settings(),
    )


def flag_button_id(idx):  # Return button id
    return flag_terms[idx] + button_id_end


def total_clicked_buttons():  # Show all clicked buttons
    return [p["prop_id"] for p in dash.callback_context.triggered][0]


def name_to_pd(name, csv_obj=None, rest_obj=None):  # Convert files to pd.DataFrames
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


def convert_to_pd(
    contents=None, filename=None, url=None
):  # Convert an uploaded table file/typed-in URL to a pd.DataFrame
    empty_df, empty_error = None, None
    error_ending = "csv, tsv, xls(x), or ods file containing a table."
    url_error = "Please enter a valid public URL to a " + error_ending
    file_error = "Please upload a valid " + error_ending
    if not contents and filename:
        try:
            df = name_to_pd(filename)
        except Exception:
            return empty_df, html_text(file_error)
        return df, empty_error
    if contents and filename:
        try:
            _, content_string = contents.split(",")
            decoded = base64.b64decode(content_string)
            df = name_to_pd(filename, io.StringIO(decoded.decode("utf-8")), io.BytesIO(decoded))
        except Exception:
            return empty_df, html_text(file_error)
        return df, empty_error
    if url:
        if validators.url(url):
            if "https://docs.google.com/spreadsheets/" in url:  # In case this is a Google Sheets URL
                url = url.replace("/edit#gid=", "/export?format=csv&gid=")
            try:
                df = name_to_pd(url)
            except Exception:
                return empty_df, html_text(url_error)
            return df, empty_error
        else:
            return empty_df, html_text(url_error)


def set_max_rows_cols(df):  # Set maximum number of rows/columns to work with for a table
    if len(df) > max_pred_rows:
        df = df.iloc[:max_pred_rows]
    if len(df.columns) > max_pred_cols:
        df = df.iloc[:, :max_pred_cols]
    return df


def show_table(
    contents=None, filename=None, url=None, table_answer=False
):  # Show a table from an uploaded file/typed-in URL
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
        items.append(html.Br())
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


def convert_to_b64(
    contents=None, filename=None, url=None
):  # Convert an uploaded image file/typed-in URL to a base64-encoded image
    empty_image, empty_error = None, None
    error_ending = "image."
    url_error = "Please enter a valid public URL to an " + error_ending
    file_error = "Please upload a valid " + error_ending
    if not contents and filename:
        try:
            image = open_image(filename)
            image = encode_b64_image(image)
        except Exception:
            return empty_image, html_text(file_error)
        return image, empty_error
    if contents and filename:
        try:
            image = contents
        except Exception:
            return empty_image, html_text(file_error)
        return image, empty_error
    if url:
        if validators.url(url):
            try:
                image = open_image(url)
                image = encode_b64_image(image)
            except Exception:
                return empty_image, html_text(url_error)
            return image, empty_error
        else:
            return empty_image, html_text(url_error)


def show_image(
    contents=None, filename=None, url=None, image_answer=False
):  # Show an image from an uploaded file/typed-in URL
    heading = ""
    image, error = None, None
    if not contents and filename:
        heading = str(filename).split("/")[-1]
        image, error = convert_to_b64(None, str(filename), None)
    if contents and filename:
        heading = filename
        image, error = convert_to_b64(contents, filename, None)
    if url:
        heading = url
        image, error = convert_to_b64(None, None, url)
    if image_answer is not None:
        image = image_answer

    if image is not None and not error:
        items = []
        if heading:
            if validators.url(heading):
                items.append(html.A(html_text("URL"), href=heading, target="_blank"))
            else:
                if len(heading) > max_char_length:
                    heading = heading[:max_char_length] + "..."
                items.append(html_text(heading))
        items.append(html.Br())
        items.append(
            html.Img(
                src=image,
                style=html_settings(width="50%"),
            ),
        )
        return html.Div(items)
    else:
        return error


def manage_output(
    code=None,
    tables=None,
    texts=None,
    graphs=None,
    images=None,
    report=None,
):  # Manage/show output
    outputs = []

    if code:
        global answers
        answers.append(code)

    if texts:
        if type(texts) == str:
            texts = [texts]
        elements = []
        for text in texts:
            elements.append(html_text(text))
            elements.append(html.Br())
        elements.append(html.Br())
        outputs.extend(
            [
                html.Center(elements),
            ]
        )

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
        elements.append(html.Br())
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
        elements.append(html.Br())
        outputs.extend(
            [
                html.Center(elements),
            ]
        )

    if images:
        elements = []
        for image in images:
            elements.append(show_image(image_answer=image))
            elements.append(html.Br())
        elements.append(html.Br())
        outputs.extend(
            [
                html.Center(elements),
            ]
        )

    if report:
        s3.download_file(Bucket=write_bucket, Key=report_name, Filename=str(asset_path / report_name))
        s3.delete_object(Bucket=write_bucket, Key=report_name)
        message, report_path = report[0], report[1]
        outputs.extend(
            [
                html.Center(
                    [
                        html_text(message),
                        html.Br(),
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


def flag_output(request, incorrect=None, offensive=None, other=None):  # Flag output
    clicked = ["temp"]
    log = pd.DataFrame(
        {
            "Request": [request],
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


# Main app code
def make_app(predict):
    # Initializing dash app
    app = dash.Dash(
        __name__,
        external_stylesheets=external_stylesheets,
        server=asset_server,
        suppress_callback_exceptions=True,
    )

    # Initializing app layout
    app.layout = html.Div(
        style=html_settings(width="100%"),
        children=[
            # Webpage title
            html.Center(
                [
                    # Intro
                    html.Div([html.Br()] * 2),
                    html_text("Captafied", "70px", "bold"),
                    html_text("Edit, query, graph, and understand your table!", "30px"),  # Subtitle
                    html.Div([html.Br()] * 2),  # Spacing
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
                    ),  # Description + usage instructions
                    html.Div([html.Br()] * 2),
                    # Input: table file
                    html_text("Upload your table:"),
                    html.Br(),
                    html.Button(
                        example_phrase,
                        id="table-file-upload-example",
                        n_clicks=0,
                        style=html_settings(width="50%"),
                    ),
                    html.Div([html.Br()] * 2),
                    dcc.Upload(
                        id="before-table-file-uploaded",
                        children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                        style=html_settings(height="400%", lineHeight="400%", borderStyle="dashed"),
                        accept=".csv, .tsv, .xls, .xlsx, .ods",
                    ),
                    html.Br(),
                    dcc.Loading(
                        type=spinner,
                        children=html.Div(id="after-table-file-uploaded"),
                    ),
                    html.Br(),
                    # Input: table as a URL
                    html_text("Or paste a public URL to it:"),
                    html.Br(),
                    html.Button(
                        example_phrase,
                        id="table-url-upload-example",
                        n_clicks=0,
                        style=html_settings(width="50%"),
                    ),
                    html.Div([html.Br()] * 2),
                    html_input(id="before-table-url-uploaded", type="url", placeholder="https://"),
                    html.Div([html.Br()] * 2),
                    dcc.Loading(
                        type=spinner,
                        children=html.Div(id="after-table-url-uploaded"),
                    ),
                    html.Div([html.Br()] * 2),
                    # Input: image file
                    html_text("Optionally, upload an image file for searching, classification, and more:"),
                    html.Br(),
                    html.Button(
                        example_phrase,
                        id="image-file-upload-example",
                        n_clicks=0,
                        style=html_settings(width="50%"),
                    ),
                    html.Div([html.Br()] * 2),
                    html.Center(
                        [
                            dcc.Upload(
                                id="before-image-file-uploaded",
                                children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
                                style=html_settings(height="400%", lineHeight="400%", borderStyle="dashed"),
                                accept=".png, .jpg, .jpeg, .webp, .gif",
                            ),
                        ]
                    ),
                    html.Br(),
                    html.Div(id="after-image-file-uploaded"),
                    html.Br(),
                    # Input: image as a URL
                    html_text("Or paste a public URL to it:"),
                    html.Br(),
                    html.Button(
                        example_phrase,
                        id="image-url-upload-example",
                        n_clicks=0,
                        style=html_settings(width="50%"),
                    ),
                    html.Div([html.Br()] * 2),
                    html_input(id="before-image-url-uploaded", type="url", placeholder="https://"),
                    html.Div([html.Br()] * 2),
                    html.Div(id="after-image-url-uploaded"),
                    html.Div([html.Br()] * 2),
                    # Input: text request
                    html_text("Ask me about your table:"),
                    dcc.Markdown(
                        [
                            """
                    [![Typing SVG](https://readme-typing-svg.demolab.com?font=Helvetica&duration=2500&pause=1000&center=true&width=350&lines=Plot+stars+over+forks.;Which+description+is+the+longest%3F;What+do+the+icons+look+like%3F;)](https://git.io/typing-svg)
                    """
                        ],
                        style=html_settings(),
                    ),  # Width = 350px for Typing SVG for iphone devices
                    html_input(id="request", type="text", placeholder="Type your request here..."),
                    html.Div([html.Br()] * 2),
                    html.Button(
                        "submit",
                        id="request-uploaded",
                        n_clicks=0,
                        style=html_settings(width="50%"),
                    ),
                    html.Div([html.Br()] * 2),
                    # Output: table, text, graph, report, and/or error
                    dcc.Loading(
                        type=spinner,
                        children=html.Div(id="pred_output"),
                    ),
                    html.Div([html.Br()] * 2),
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
                    html.Div([html.Br()] * 2),
                    dcc.Loading(
                        type=spinner,
                        children=html.Div(id="flag_output"),
                    ),
                    # Extra length at the bottom of the page
                    html.Div([html.Br()] * 5),
                ]
            ),
        ],
    )

    # Event functions
    @app.callback(
        Output("after-table-file-uploaded", "children"),
        Input("before-table-file-uploaded", "contents"),
        State("before-table-file-uploaded", "filename"),
        Input("table-file-upload-example", "n_clicks"),
    )
    def show_uploaded_table_file(contents, filename, example_clicked):  # When only table file is uploaded
        if "table-file-upload-example" in total_clicked_buttons():
            return show_table(None, table_file_example, None, None)
        if contents and filename:
            if type(contents) == list and type(filename) == list:
                if len(contents) > 1 and len(filename) > 1:
                    return html_text(multiple_files_error)
            return show_table(contents, filename, None, None)

    @app.callback(
        Output("after-table-url-uploaded", "children"),
        Input("before-table-url-uploaded", "value"),
        Input("table-url-upload-example", "n_clicks"),
    )
    def show_uploaded_table_url(url, example_clicked):  # Show example/uploaded table URL
        if "table-url-upload-example" in total_clicked_buttons():
            return show_table(None, None, table_url_example, None)
        if url:
            return show_table(None, None, url, None)

    @app.callback(
        Output("after-image-file-uploaded", "children"),
        Input("before-image-file-uploaded", "contents"),
        State("before-image-file-uploaded", "filename"),
        Input("image-file-upload-example", "n_clicks"),
    )
    def show_uploaded_image_file(contents, filename, example_clicked=None):  # Show example/uploaded image file
        if "image-file-upload-example" in total_clicked_buttons():
            return show_image(None, image_file_example, None, None)
        if contents and filename:
            if type(contents) == list and type(filename) == list:
                if len(contents) > 1 and len(filename) > 1:
                    return html_text(multiple_files_error)
            return show_image(contents, filename, None, None)

    @app.callback(
        Output("after-image-url-uploaded", "children"),
        Input("before-image-url-uploaded", "value"),
        Input("image-url-upload-example", "n_clicks"),
    )
    def show_uploaded_image_url(url, example_clicked):  # Show example/uploaded image url
        if "image-url-upload-example" in total_clicked_buttons():
            return show_image(None, None, image_url_example, None)
        if url:
            return show_image(None, None, url, None)

    @app.callback(
        Output("pred_output", "children"),
        State("after-table-file-uploaded", "children"),
        State("after-table-url-uploaded", "children"),
        State("after-image-file-uploaded", "children"),
        State("after-image-url-uploaded", "children"),
        Input("request-uploaded", "n_clicks"),
        Input("request", "value"),
        Input("before-table-file-uploaded", "contents"),
        State("before-table-file-uploaded", "filename"),
        Input("before-table-url-uploaded", "value"),
        Input("before-image-file-uploaded", "contents"),
        State("before-image-file-uploaded", "filename"),
        Input("before-image-url-uploaded", "value"),
    )
    def get_prediction(
        show_table_file,
        show_table_url,
        show_image_file,
        show_image_url,
        submit,
        request,
        table_contents,
        table_filename,
        table_url,
        image_contents,
        image_filename,
        image_url,
    ):  # When table file/url + request +/- image are uploaded
        if "request-uploaded" in total_clicked_buttons():
            df, error = None, None
            table_checks = [
                table_contents and table_filename and request,
                table_url and request,
                show_table_file and request,
                show_table_url and request,
            ]
            image_checks = [
                image_contents and image_filename,
                image_url,
                show_image_file,
                show_image_url,
            ]
            if table_checks[0]:
                df, error = convert_to_pd(table_contents[0], table_filename[0], None)
            if table_checks[1]:
                df, error = convert_to_pd(None, None, table_url)
            if (not table_checks[0] and not table_checks[1]) and table_checks[2]:
                df, error = convert_to_pd(None, str(table_file_example), None)
            if (not table_checks[0] and not table_checks[1]) and table_checks[3]:
                df, error = convert_to_pd(None, None, str(table_url_example))
            if df is not None and not error:
                global output_tables
                output_tables = []
                df = set_max_rows_cols(df)

                image = None
                if image_checks[0]:
                    image, error = convert_to_b64(image_contents[0], image_filename[0], None)
                if image_checks[1]:
                    image, error = convert_to_b64(None, None, image_url)
                if (not image_checks[0] and not image_checks[1]) and image_checks[2]:
                    image, error = convert_to_b64(None, str(image_file_example), None)
                if (not image_checks[0] and not image_checks[1]) and image_checks[3]:
                    image, error = convert_to_b64(None, None, str(image_url_example))
                if not error:
                    global requests
                    requests.append({"text": request, "image": image})
                    code, tables, texts, graphs, images, report = predict(df, requests, answers)
                    if tables:
                        tables = [pd.DataFrame.from_dict(table) for table in tables]
                    if graphs:
                        graphs = [pio.from_json(graph) for graph in graphs]
                    return manage_output(code, tables, texts, graphs, images, report)

    @app.callback(
        Output("download-table-csv", "data"),
        Input("download-table" + button_id_end, "n_clicks"),
    )
    def download_table(n_clicks):  # When download button is clicked
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

    @app.callback(
        Output("flag_output", "children"),
        State("after-table-file-uploaded", "children"),
        State("after-table-url-uploaded", "children"),
        State("request", "value"),
        State("before-table-file-uploaded", "contents"),
        State("before-table-file-uploaded", "filename"),
        State("before-table-url-uploaded", "value"),
        Input("incorrect-button-state", "n_clicks"),
        Input("offensive-button-state", "n_clicks"),
        Input("other-button-state", "n_clicks"),
    )
    def flag_pred(
        show_file, show_url, request, contents, filename, url, inc_clicked, off_clicked, oth_clicked
    ):  # When flagging button(s) is/are clicked
        changed_id = total_clicked_buttons()
        buttons_clicked = [
            flag_button_id(0) in changed_id,
            flag_button_id(1) in changed_id,
            flag_button_id(2) in changed_id,
        ]

        table_checks = [
            True in buttons_clicked,
            contents and filename and request,
            url and request,
            show_file and request,
            show_url and request,
        ]
        df, error = None, None
        if table_checks[0] and table_checks[1]:
            df, error = convert_to_pd(contents[0], filename[0], None)
        if table_checks[0] and table_checks[2]:
            df, error = convert_to_pd(None, None, url)
        if table_checks[0] and (not table_checks[1] and not table_checks[2]) and table_checks[3]:
            df, error = convert_to_pd(None, str(table_file_example), None)
        if table_checks[0] and (not table_checks[1] and not table_checks[2]) and table_checks[4]:
            df, error = convert_to_pd(None, None, str(table_url_example))

        check = (show_file or show_url) and df is not None and request and not error
        if check and buttons_clicked[0]:
            return flag_output(request, True, None, None)
        if check and buttons_clicked[1]:
            return flag_output(request, None, True, None)
        if check and buttons_clicked[2]:
            return flag_output(request, None, None, True)

    @app.server.route("/assets/<resource>")
    def serve_assets(resource):  # When report is generated and needs to be displayed
        return flask.send_from_directory(ASSETS_PATH, resource)

    return app


# Main backend code
class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL, provide the url kwarg.

    Otherwise, runs a predictor locally.
    """

    def __init__(self, use_url):
        if use_url:
            self.url = BACKEND_URL
            self._predict = self._predict_from_endpoint
        # Uncomment this to run the backend locally
        # else:
        #     from backend.inference.inference import Pipeline  # so that we don't have to install backend as a dependency

        #     model = Pipeline()
        #     self._predict = model.predict

    def run(self, df, requests, answers):
        pred = self._predict(df, requests, answers)
        self._log_inference(pred)
        return pred

    def _predict_from_endpoint(self, df, requests, answers):
        headers = {"Content-type": "application/json"}
        payload = json.dumps(
            {
                "table": df.to_dict(),
                "requests": requests,
                "prev_answers": answers,
            }
        )

        try:
            response = req.post(self.url, data=payload, headers=headers)
            pred = response.json()["pred"]
        except Exception:
            pred = [None, None, "Sorry, something went wrong. Please try again.", None, None, None]

        return pred

    def _log_inference(self, pred):
        logging.info(f"PRED >begin\n{pred}\nPRED >end")


predictor = PredictorBackend(use_url=True)
app = make_app(predictor.run)


if __name__ == "__main__":
    waitress.serve(app.server, port=DEFAULT_PORT)
