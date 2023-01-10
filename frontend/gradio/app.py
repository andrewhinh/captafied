"""Edit or ask any question about your table!"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable

import gradio as gr
import pandas as pd
import requests

from backend.inference import util
from backend.inference.inference import Pipeline


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)

APP_DIR = Path(__file__).resolve().parent  # what is the directory for this application?
FRONTEND_DIR = Path("./frontend")  # what is the directory for the frontend?
FAVICON = FRONTEND_DIR / "logo.png"  # path to a small image for display in browser tab and social media
README = APP_DIR / "README.md"  # path to an app readme file in HTML/markdown

DEFAULT_PORT = 11700

test_path = Path("backend") / "inference" / "tests" / "support"


def get_examples(folder, ext, table_paths=None):
    ex_dir = test_path / folder
    ex_fnames = [elem for elem in os.listdir(ex_dir) if elem.endswith(ext)]
    ex_paths = [ex_dir / fname for fname in ex_fnames]
    ex_paths = sorted(ex_paths)

    if table_paths:
        requests = []
        for path in ex_paths:
            with open(path, "r") as f:
                requests.append(f.readline())
        if len(table_paths) > 1:
            temp = []
            for table_path in table_paths:
                temp.extend([table_path] * len(requests))
            requests *= len(temp)
        else:
            temp = table_paths.copy()
            temp *= len(requests)
        return [[str(table_path), request] for table_path, request in zip(temp, requests)]
    else:
        return ex_paths


def get_interface(fn, outputs, readme, examples, allow_flagging, flagging_callback, flagging_dir):
    return gr.Interface(
        fn=fn,  # which Python function are we interacting with?
        outputs=outputs, # what output widgets does it need?
        # what input widgets does it need?
        inputs=[gr.components.File(label="Table"), gr.components.Textbox(label="Request")],
        title="Captafied",  # what should we display at the top of the page?
        thumbnail=FAVICON,  # what should we display when the link is shared, e.g. on social media?
        description=__doc__,  # what should we display just above the interface?
        article=readme,  # what long-form content should we display below the interface?
        examples=examples,  # which potential inputs should we provide?
        cache_examples=False,  # should we cache those inputs for faster inference? slows down start
        allow_flagging=allow_flagging,  # should we show users the option to "flag" outputs?
        flagging_options=["incorrect", "offensive", "other"],  # what options do users have for feedback?
        flagging_callback=flagging_callback,
        flagging_dir=flagging_dir,
    )


def main(args):
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(predictor.run, flagging=args.flagging)
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=args.port,  # set a port to bind to, failing if unavailable
        share=True,  # should we create a (temporary) public link on https://gradio.app?
        favicon_path=FAVICON,  # what icon should we display in the address bar?
    )


def make_frontend(fn: Callable[[pd.DataFrame, str], str], flagging: bool = False):
    """Creates a gradio.Interface frontend for an table + text to table, text, graph, or HTML page function."""
    table_example_paths = get_examples("tables", ".csv")
    table_examples = get_examples("table_requests", ".txt", table_example_paths)
    text_examples = get_examples("text_requests", ".txt", table_example_paths)
    graph_examples = get_examples("graph_requests", ".txt", table_example_paths)
    report_examples = get_examples("report_requests", ".txt", table_example_paths)

    allow_flagging = "never"
    if flagging:  # logging user feedback to a local CSV file
        allow_flagging = "manual"
        flagging_callback = gr.CSVLogger()
        flagging_dir = "flagged"
    else:
        flagging_callback, flagging_dir = None, None

    readme = _load_readme(with_logging=allow_flagging == "manual")

    # build a basic browser interface to a Python function
    table = get_interface(
        fn=fn,
        outputs=[gr.components.DataFrame()],
        readme=readme,
        examples=table_examples,
        allow_flagging=allow_flagging,
        flagging_callback=flagging_callback,
        flagging_dir=flagging_dir,
    )

    text = get_interface(
        fn=fn,
        outputs=[gr.components.Textbox()],
        readme=readme,
        examples=text_examples,
        allow_flagging=allow_flagging,
        flagging_callback=flagging_callback,
        flagging_dir=flagging_dir,
    )

    graph = get_interface(
        fn=fn,
        outputs=[gr.components.Plot()],
        readme=readme,
        examples=graph_examples,
        allow_flagging=allow_flagging,
        flagging_callback=flagging_callback,
        flagging_dir=flagging_dir,
    )

    report = get_interface(
        fn=fn,
        outputs=[gr.components.HTML()],
        readme=readme,
        examples=report_examples,
        allow_flagging=allow_flagging,
        flagging_callback=flagging_callback,
        flagging_dir=flagging_dir,
    )

    frontend = gr.TabbedInterface(interface_list=[table, text, graph, report],
                                  tab_names=["Table", "Text", "Graph", "Report"],
    )

    return frontend


class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL, provide the url kwarg.

    Otherwise, runs a predictor locally.
    """

    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = Pipeline()
            self._predict = model.predict

    def run(self, table, request):
        pred = self._predict_with_metrics(table, request) #, metrics
        self._log_inference(pred) #, metrics
        return pred

    def _predict_with_metrics(self, table, request):
        pred = self._predict(table, request)
        """
        stats = ImageStat.Stat(image)
        metrics = {
            "image_mean_intensity": stats.mean,
            "image_median": stats.median,
            "image_extrema": stats.extrema,
            "image_area": image.size[0] * image.size[1],
            "pred_length": len(pred),
        }
        """
        return pred#, metrics

    def _predict_from_endpoint(self, table, request):
        """Send an table and request to an endpoint that accepts JSON and return the predictions.

        The endpoint should expect a base64 representation of the image, encoded as a string,
        under the key "image" and a str representation of the request. It should return the predicted text under the key "pred".

        Parameters
        ----------
        image
            A PIL image of handwritten text to be converted into a string

        request
            A string containing the user's request

        Returns
        -------
        pred
            A string containing the predictor's guess of the text in the image.
        """
        encoded_image = util.encode_b64_image(image)

        headers = {"Content-type": "application/json"}
        payload = json.dumps(
            {"image": "data:image/jpg;base64," + encoded_image, "request": "data:request/str;str," + request}
        )

        response = requests.post(self.url, data=payload, headers=headers)
        pred = response.json()["pred"]

        return pred

    def _log_inference(self, pred): #, metrics
        #for key, value in metrics.items():
            #logging.info(f"METRIC {key} {value}")
        logging.info(f"PRED >begin\n{pred}\nPRED >end")


def _load_readme(with_logging=False):
    with open(README) as f:
        lines = f.readlines()
        if not with_logging:
            lines = lines[: lines.index("<!-- logging content below -->\n")]

        readme = "".join(lines)
    return readme


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Identifies a URL to which to send image data. Data is base64-encoded, converted to a utf-8 string, and then set via a POST request as JSON with the key 'image'. Default is None, which instead sends the data to a model running locally.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--flagging",
        action="store_true",
        help="Pass this flag to allow users to 'flag' model behavior and provide feedback.",
    )

    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
