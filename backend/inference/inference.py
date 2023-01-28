# Imports
import base64
from functools import partial
from io import BytesIO
import itertools
import os
from os import path
from pathlib import Path
import traceback
from typing import List, Optional

from dotenv import load_dotenv
import numpy as np
from onnxruntime import InferenceSession
import openai
from openai.embeddings_utils import cosine_similarity
import pandas as pd
from pandas_profiling import ProfileReport
from PIL import Image
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import requests as rq
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import CLIPProcessor
import umap
import validators


# Setup
# Plotly setup
pio.templates.default = "plotly_dark"

# Loading env variables
load_dotenv()

# OpenAI API setup
openai.organization = "org-SenjN6vfnkIealcfn6t9JOJj"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Artifacts (models, etc.) path
parent_path = Path(__file__).resolve().parent
artifact_path = parent_path / "artifacts" / "inference"
onnx_path = artifact_path / "onnx"

# CLIP encoders config
clip_processor = artifact_path / "clip-vit-base-patch16"
clip_onnx = onnx_path / "clip.onnx"


# Classes
class InvalidRequest(ValueError):
    """Raise this when an invalid request is made to the API"""


class Pipeline:
    """
    Main inference class
    """

    def __init__(self):
        # CLIP Setup
        self.clip_session = InferenceSession(str(clip_onnx))
        self.clip_processor = CLIPProcessor.from_pretrained(clip_processor)

        # OpenAI Engine
        self.engine = "text-davinci-003"
        self.max_prompt_tokens = 4000
        self.max_manual_tokens = 50
        self.max_code_tokens = 350
        self.max_prev = int(self.max_prompt_tokens / (self.max_code_tokens + self.max_manual_tokens)) - 2  # buffer

        # OpenAI Context
        self.data_types = ["text string", "image path/URL", "categorical", "continuous"]

        # Number of output types
        self.outputs = ["code", "table", "text", "plot", "image", "report"]
        self.num_outputs = len(self.outputs)

        # Error messages
        self.invalid_request_error = "I don't know how to answer that question. "
        self.other_error = "Something went wrong. "
        self.report_message = (
            "Here's a report on the table generated by YData's pandas-profiling library that might help you."
        )

    def is_string_series(self, s: pd.Series):
        if isinstance(s.dtype, pd.StringDtype):
            # The series was explicitly created as a string series (Pandas>=1.0.0)
            return True
        elif s.dtype == "object":
            # Object series, check each value
            return all((v is None) or isinstance(v, str) for v in s)
        else:
            return False

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def openai_query(self, **kwargs):
        response = openai.Completion.create(
            engine=self.engine,
            **kwargs,
        )
        return response["choices"][0]["text"].strip()

    def open_image(self, image):
        image_pil = Image.open(image)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        return np.array(image_pil)

    def img_to_str(self, image):
        pil_img = Image.fromarray(image)
        buff = BytesIO()
        pil_img.save(buff, format="PNG")
        img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
        return "data:image/png;base64," + img_str

    def exec_code(self, global_vars, code):
        result = []
        local_vars = {"result": result}
        try:
            exec(code, global_vars, local_vars)
            return local_vars["result"]
        except Exception:
            print(traceback.format_exc())
            return None

    def get_column_vals(self, table, column, images_present=False):
        if images_present:
            return [self.open_image(image) for image in table[column]]
        else:
            return list(table[column])

    def clip_encode(self, texts, images):
        # Set up inputs for CLIP
        if not texts:
            clip_images = list(itertools.chain.from_iterable(images))
            clip_texts = ["Placeholder" for _ in range(len(clip_images))]
        elif not images:
            clip_texts = list(itertools.chain.from_iterable(texts))
            clip_images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(len(clip_texts))]
        else:
            clip_texts = list(itertools.chain.from_iterable(texts))
            clip_images = list(itertools.chain.from_iterable(images))
            if len(clip_texts) > len(clip_images):
                clip_images *= len(clip_texts) // len(clip_images)
            elif len(clip_images) > len(clip_texts):
                clip_texts *= len(clip_images) // len(clip_texts)
            else:
                pass

        # CLIP encoding
        inputs = self.clip_processor(text=clip_texts, images=clip_images, return_tensors="np", padding=True)
        clip_outputs = self.clip_session.run(
            output_names=["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"], input_feed=dict(inputs)
        )
        text_embeds = clip_outputs[2]
        image_embeds = clip_outputs[3]
        return text_embeds, image_embeds

    def get_embeds_graph(self, max_cluster_search, text_image_cols, text_image_embeds, dict_cont, dict_cat):
        # Conditionals for plotting
        one_cont_var_present = len(dict_cont) == 1
        two_cont_vars_present = len(dict_cont) == 2
        two_or_more_text_image = len(text_image_cols) > 1

        # Setting up figure
        fig = make_subplots()
        if dict_cont:  # If there are continuous columns, prepare for 3D plot
            list_markers = ["circle", "circle-open", "cross", "diamond", "diamond-open", "square", "square-open", "x"]
        else:
            list_markers = (
                list(range(0, 55))
                + list(range(100, 155))
                + list(range(200, 225))
                + [236]
                + list(range(300, 325))
                + [326]
            )  # https://plotly.com/python/marker-style/#custom-marker-symbols
        markers = itertools.cycle((list_markers))
        color_themes = (
            px.colors.qualitative
        )  # https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
        list_colors = [getattr(color_themes, att) for att in dir(color_themes)[:-11]]
        colors = list(itertools.chain.from_iterable(list_colors))

        # UMAP and K-Means clustering
        for column, embeds, color in zip(text_image_cols, text_image_embeds, colors):
            # Legend entries depending on whether there are multiple text/image columns
            prefix = ""
            if two_or_more_text_image:
                prefix = column + ", "

            # Getting # of clusters and K-Means clustering
            range_n_clusters = list(
                range(2, max_cluster_search)
            )  # For performance reasons, we don't want to cluster more than half the data
            silhouette_scores = []
            for num_clusters in range_n_clusters:
                # initialise kmeans
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(embeds)
                labels = kmeans.labels_
                # silhouette score
                silhouette_scores.append(silhouette_score(embeds, labels))
            n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
            kmeans.fit(embeds)
            clusters = kmeans.labels_

            # Extending added variables to match the number of data points
            if dict_cat:
                for item in dict_cat.items():
                    dict_cat[item[0]] = item[1] * int(len(clusters) / len(item[1]))
            if dict_cont:
                for item in dict_cont.items():
                    dict_cont[item[0]] = item[1] * int(len(clusters) / len(item[1]))

            # Collecting min/max values of continuous variables for axis ranges
            min_x = []
            max_x = []
            min_y = []
            max_y = []
            min_z = []
            max_z = []

            # Reducing dimensionality of embeddings with UMAP
            n_neighbors = 15
            n_components = 2
            if embeds.shape[0] < 15:  # UMAP's default n_neighbors=15, reduce if # of data points is less than 15
                n_neighbors = embeds.shape[0] - 1
            if two_cont_vars_present:  # Reduce UMAP n_components to accomodate for 2 cont variables
                n_components = 1
            reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
            embedding = reducer.fit_transform(embeds)
            if n_components == 1:
                x = embedding[:, 0]
                y = None
            else:
                x = embedding[:, 0]
                y = embedding[:, 1]

            # Plotting clusters
            for cluster in range(n_clusters):
                # Getting vars for scatter plot
                marker = next(markers)
                scatter = partial(go.Scatter, mode="markers", marker=dict(symbol=marker, color=color))
                scatter_3d = partial(go.Scatter3d, mode="markers", marker=dict(symbol=marker, color=color))

                # Text/image data
                xs = np.array(x)[clusters == cluster]
                min_x.append(min(xs))
                max_x.append(max(xs))

                if dict_cat and (
                    not dict_cont or one_cont_var_present
                ):  # Only categorical variables or (1 continuous variable and 1+ categorical variables)
                    ys = np.array(y)[clusters == cluster]  # text/image data
                    min_y.append(min(ys))
                    max_y.append(max(ys))
                    if one_cont_var_present:
                        zs = np.array(list(dict_cont.values())[0])[clusters == cluster]
                        min_z.append(min(zs))
                        max_z.append(max(zs))
                    total_cats = [list(np.array(cats)[clusters == cluster]) for cats in list(dict_cat.values())]
                    unique_cats = [np.unique(cats) for cats in total_cats]
                    cat_combinations = list(itertools.product(*unique_cats))
                    total_cats = np.array(total_cats[0])
                    for cat_combination in cat_combinations:
                        name = (
                            prefix
                            + ", ".join(
                                [list(dict_cat.keys())[i] + ": " + str(cat) for i, cat in enumerate(cat_combination)]
                            )
                            + ", "
                            + "Cluster: "
                            + str(cluster)
                        )
                        temp_x = xs[total_cats == cat_combination]
                        temp_y = ys[total_cats == cat_combination]
                        if one_cont_var_present:
                            temp_z = zs[total_cats == cat_combination]
                            fig.add_trace(
                                scatter_3d(
                                    x=temp_x,
                                    y=temp_y,
                                    z=temp_z,
                                    name=name,
                                    marker=dict(symbol=marker, color=color),
                                ),
                            )
                        else:
                            fig.add_trace(
                                scatter(
                                    x=temp_x,
                                    y=temp_y,
                                    name=name,
                                    marker=dict(symbol=marker, color=color),
                                ),
                            )

                        marker = next(markers)

                elif one_cont_var_present:  # Only 1 continuous variable
                    ys = np.array(y)[clusters == cluster]
                    zs = np.array(list(dict_cont.values())[0])[clusters == cluster]
                    min_y.append(min(ys))
                    max_y.append(max(ys))
                    min_z.append(min(zs))
                    max_z.append(max(zs))
                    name = prefix + "Cluster: " + str(cluster)
                    fig.add_trace(scatter_3d(x=xs, y=ys, z=zs, name=name))

                elif two_cont_vars_present:  # Only 2 continuous variables
                    ys = np.array(list(dict_cont.values())[0])[clusters == cluster]
                    zs = np.array(list(dict_cont.values())[1])[clusters == cluster]
                    min_y.append(min(ys))
                    max_y.append(max(ys))
                    min_z.append(min(zs))
                    max_z.append(max(zs))
                    name = prefix + "Cluster: " + str(cluster)
                    fig.add_trace(scatter_3d(x=xs, y=ys, z=zs, name=name))

                else:  # No continuous/categorical variables
                    ys = np.array(y)[clusters == cluster]
                    min_y.append(min(ys))
                    max_y.append(max(ys))
                    name = prefix + "Cluster: " + str(cluster)
                    fig.add_trace(scatter(x=xs, y=ys, name=name))

            # Labelling + setting axis ranges
            if dict_cont:
                # Getting ranges for axis
                x_min = min(min_x)
                x_max = max(max_x)
                if type(x_min) == np.ndarray and type(x_max) == np.ndarray:
                    x_min = x_min[0]
                    x_max = x_max[0]
                y_min = min(min_y)
                y_max = max(max_y)
                z_min = min(min_z)
                z_max = max(max_z)

                # Setting axis ranges
                if one_cont_var_present:
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(
                                range=[x_min, x_max],
                            ),
                            yaxis=dict(
                                range=[y_min, y_max],
                            ),
                            zaxis=dict(
                                title=list(dict_cont.keys())[0],
                                range=[z_min, z_max],
                            ),
                        )
                    )
                elif two_cont_vars_present:
                    fig.update_layout(
                        scene=dict(
                            xaxis=dict(
                                range=[x_min, x_max],
                            ),
                            yaxis=dict(
                                title=list(dict_cont.keys())[0],
                                range=[y_min, y_max],
                            ),
                            zaxis=dict(
                                title=list(dict_cont.keys())[1],
                                range=[z_min, z_max],
                            ),
                        )
                    )

        """
        # HDBSCAN clustering
        import hdbscan
        offset = 0 # To label the clusters in a continuous manner
        for embeds, color in zip(list_embeds, colors):
            # Reducing dimensionality of embeddings with UMAP
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(embeds)
            x = embedding[:, 0]
            y = embedding[:, 1]

            clusters = hdbscan.HDBSCAN().fit_predict(embedding)
            clustered = (clusters >= 0)
            n_clusters = max(clusters) + 1

            clustered_x = embedding[clustered, 0]
            clustered_y = embedding[clustered, 1]
            unclustered_x = embedding[~clustered, 0]
            unclustered_y = embedding[~clustered, 1]
            groups = [[clustered_x, clustered_y], [unclustered_x, unclustered_y]]

            # Plotting clusters and unclustered points
            for group_idx in range(len(groups)):
                group = groups[group_idx]
                for cluster in range(n_clusters):
                    # Plotting points
                    xs = np.array(group[0])[clusters == cluster]
                    ys = np.array(group[1])[clusters == cluster]
                    if group_idx < len(groups) - 1:
                        # Using same marker for all unclustered points
                        marker = "."
                        # Adding cluster to legend if applicable
                        artist = matplotlib.lines.Line2D([], [], color=color, lw=0, marker=marker)
                        handles.append(artist)
                        labels.append(str(cluster + offset))
                    else:
                        marker = next(markers)
                    ax.scatter(xs, ys, color=color, marker=marker, alpha=1)

            # To label the clusters in a continuous manner
            if len(list_embeds) > 1:
                offset += n_clusters
        # Legend for clusters
        legend = ax.legend(handles, labels, loc="upper right", title="Clusters")
        ax.add_artist(legend)

        # Adding legend for image and text groups
        if len(list_embeds) > 1:
            handles = []
            labels = []
            for color, type in zip(colors, self.data_types):
                artist = matplotlib.lines.Line2D([], [], color=color, lw=0, marker="o")
                handles.append(artist)
                labels.append(type[0].upper() + type[1:])
            ax.legend(handles, labels, loc="lower left", title="Data Types")
        """

        # Titles
        variables = text_image_cols + list(dict_cat.keys()) + list(dict_cont.keys())
        variables = [
            variable + " Embedding Clusters" if variable in text_image_cols else variable for variable in variables
        ]
        variables = " vs. ".join(variables)
        legend_titles = ["Embedding Clusters", "Columns", "Categories"]
        if two_or_more_text_image and dict_cat:  # 2 or more text/image columns and 1 or more categorical columns
            legend_title = ", ".join(legend_titles)
        elif two_or_more_text_image:  # 2 or more text/image columns
            legend_title = " and ".join(legend_titles[:2])
        elif dict_cat:  # 1 or more categorical columns
            legend_title = " and ".join([legend_titles[0], legend_titles[2]])
        else:  # No categorical columns and 1 text/image column
            legend_title = legend_titles[0]
        fig.update_layout(
            title={"text": variables, "y": 0.9, "x": 0.5, "xanchor": "center", "yanchor": "top"},
            legend={"title": legend_title, "y": 0.9, "x": 0.9, "xanchor": "right", "yanchor": "top"},
        )

        return fig

    def get_most_similar(self, table, cols, search_embed, embeds):
        highest_similarity = 0
        text = ""
        for text_col, text_embed in zip(cols, embeds):
            for row_idx in range(len(text_embed)):
                similarity = cosine_similarity(text_embed[row_idx], search_embed)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    text = table.loc[row_idx, text_col]
        return text

    def get_anomaly_rows(self, text_image_embeds):
        raise InvalidRequest()

    def get_diversity_measure(self, text_image_embeds):
        raise InvalidRequest()

    def get_classification_label(self, embed, text_image_embeds, dict_cat):
        raise InvalidRequest()

    def get_report(self, table, message):
        if len(table) > 1000:
            report = ProfileReport(table, title="Pandas Profiling Report", minimal=True)
        else:
            report = ProfileReport(table, title="Pandas Profiling Report", explorative=True)

        # For Dash
        report_path = Path("assets") / "report.html"
        full_report_path = parent_path / ".." / ".." / "frontend" / report_path
        report.to_file(full_report_path)
        return [message, "/" + str(report_path)]

    def predict(
        self,
        table: pd.DataFrame,
        requests: List[str],
        prev_answers: Optional[List[str]] = None,
        request_types: Optional[List[str]] = None,
    ) -> str:  # Type handling is done by frontend
        try:
            # Initializing output variables
            outputs = [[] for _ in range(self.num_outputs)]

            # Getting data types of columns for all tables
            column_data = {}
            for col_idx in range(len(table.columns)):  # For each column
                column = table.columns[col_idx]  # Get the column name
                test = table[column]  # Get the column values
                if len(set(test)) >= len(table):  # For continuous data
                    if self.is_string_series(test):  # For strings
                        for row_idx in range(len(test)):  # For each row
                            value = test[row_idx]  # Get the value
                            potential_path = str(parent_path / value)  # Get the potential path
                            if validators.url(value):  # For images
                                table.at[row_idx, column] = rq.get(value, stream=True).raw
                                if column not in column_data:
                                    column_data[column] = self.data_types[1]
                            elif path.exists(potential_path):  # For local images
                                table.at[row_idx, column] = potential_path
                                if column not in column_data:
                                    column_data[column] = self.data_types[1]
                            else:  # For text
                                column_data[column] = self.data_types[0]
                                break  # Break out of the loop at the first text value
                    else:  # For continuous data
                        column_data[column] = self.data_types[3]
                else:  # For categorical data
                    column_data[column] = self.data_types[2]

            # Introduction for every OpenAI prompt
            str_info = ", ".join([column + ": " + data_type for column, data_type in column_data.items()])
            intro = str(
                "You are given a Python pandas DataFrame named table. "
                + "You are also given a comma-separated list that contains "
                + "pairs of table's columns and corresponding data types: "
                + str_info
                + "\n"
                + "Regarding table, a user named USER "
            )
            if prev_answers:  # If there are previous requests
                if len(prev_answers) > self.max_prev:  # If there are too many previous requests for OpenAI
                    requests = requests[-self.max_prev :]
                    prev_answers = prev_answers[-self.max_prev :]
                intro += "previously asked: " + requests[0] + "\n"
                for idx, (request, prev_answer) in enumerate(zip(requests[1:], prev_answers)):
                    intro += "You answered: " + prev_answer + "\nThen, USER "
                    if idx < len(requests) - 1:  # If there are more requests
                        intro += "asked: " + request + "\n"
                    else:  # If this is the most recent one
                        intro += "now asks: " + request + "\n"
                intro += "For future instructions, consider every request and answer given.\n"
            else:  # If there are no previous requests
                intro += "asks: " + requests[0] + "\n"

            # Check if USER wants to use a manually-defined function
            if request_types:  # If so
                # Helper functions
                def select_columns():  # When USER wants to select columns
                    # Get the column names to use
                    str_cols = self.openai_query(
                        prompt=intro
                        + "Return a comma-separated list that contains the columns in table "
                        + "that USER explicitly references. Some notes:\n"
                        + "1) If USER references a past request or answer, be sure to include any past columns "
                        + "that are necessary for the current request.\n"
                        + "2) If you think a column is necessary but it is phrased in a way that suggests it posseses "
                        + "another column, it should be ignored. For example, if USER asks 'Show the repo's "
                        + "description clusters.' regarding table, the column 'Repos' should be "
                        + "ignored because it posseses the column 'Description', and only the column 'Description' "
                        + "should be used.\n"
                        + "3) If USER is asking to graph more than two categorical and/or continuous columns, "
                        + "return 'None'. ",
                        temperature=0,
                        max_tokens=self.max_manual_tokens,
                    )
                    print(str_cols + "\n\n\n\n\n")

                    # Check if the columns are valid
                    columns = []
                    for col in table.columns:
                        if col in str_cols:
                            columns.append(col)

                    # Add selected columns to the list of outputs
                    outputs[0] = str_cols

                    return columns

                def get_list(type, columns=None):
                    if columns:
                        return [k for k, v in column_data.items() if k in columns and v == type]
                    else:
                        return [k for k, v in column_data.items() if v == type]

                def get_vals_as_list(cols, image_present=None):
                    return [self.get_column_vals(table, column, image_present) for column in cols if cols]

                def get_vals_as_dict(cols):
                    return {column: self.get_column_vals(table, column) for column in cols if cols}

                def to_list_of_lists(x):
                    return [x[i : i + len(table)] for i in range(0, len(x), len(table))]

                def get_all(columns=None):
                    # CLIP embeddings of text and/or image columns
                    text_columns = get_list(self.data_types[0], columns)
                    image_columns = get_list(self.data_types[1], columns)
                    texts = get_vals_as_list(text_columns)
                    images = get_vals_as_list(image_columns, images_present=True)
                    if not texts and not images:
                        raise ValueError()
                    text_embeds, image_embeds = self.clip_encode(texts, images)
                    text_embeds = to_list_of_lists(text_embeds)
                    image_embeds = to_list_of_lists(image_embeds)

                    # Get values of categorical/continuous columns
                    dict_cat = {}
                    dict_cont = {}
                    cat_columns = get_list(self.data_types[2], columns)
                    cont_columns = get_list(self.data_types[3], columns)
                    dict_cat = get_vals_as_dict(cat_columns)
                    dict_cont = get_vals_as_dict(cont_columns)

                    return text_columns, image_columns, text_embeds, image_embeds, dict_cat, dict_cont

                if "cluster" in request_types:  # If USER wants to cluster
                    columns = select_columns()
                    if columns:
                        get_all = partial(get_all, columns)
                    text_cols, image_cols, text_embeds, image_embeds, dict_cat, dict_cont = get_all()
                    outputs[3].append(
                        self.get_embeds_graph(
                            int(len(table) / 2),
                            text_cols + image_cols,
                            text_embeds + image_embeds,
                            dict_cont,
                            dict_cat,
                        )
                    )
                elif "text_search" in request_types:  # If USER wants to search for text
                    _, image_embed = self.clip_encode([], [request])
                    text_cols, _, text_embeds, _, _, _ = get_all()
                    text = self.get_most_similar(table, text_cols, image_embed, text_embeds)
                    outputs[2].append(text)
                elif "image_search" in request_types:  # If USER wants to search for images
                    text_embed, _ = self.clip_encode([request], [])
                    _, image_cols, _, image_embeds, _, _ = get_all()
                    image = self.get_most_similar(table, image_cols, text_embed, image_embeds)
                    outputs[4].append(self.open_image(image))
                elif "anomaly" in request_types:  # If USER wants to detect anomalies
                    columns = select_columns()
                    if columns:
                        get_all = partial(get_all, columns)
                    _, _, text_embeds, image_embeds, _, _ = get_all()
                    outputs[1].append(self.get_anomaly_rows(text_embeds + image_embeds))
                elif "diversity" in request_types:  # If USER wants to measure diversity
                    columns = select_columns()
                    if columns:
                        get_all = partial(get_all, columns)
                    _, _, text_embeds, image_embeds, _, _ = get_all()
                    outputs[2].append(self.get_diversity_measure(text_embeds + image_embeds))
                elif "text_class" in request_types:  # If USER wants to classify text
                    columns = select_columns()
                    if columns:
                        get_all = partial(get_all, columns)
                    text_embed, _ = self.clip_encode([request], [])
                    _, _, text_embeds, _, dict_cat, _ = get_all()
                    outputs[2].append(self.get_classification_label(text_embed, text_embeds, dict_cat))
                elif "image_class" in request_types:  # If USER wants to classify images
                    columns = select_columns()
                    if columns:
                        get_all = partial(get_all, columns)
                    _, image_embed = self.clip_encode([], [request])
                    _, _, _, image_embeds, dict_cat, _ = get_all()
                    outputs[4].append(self.get_classification_label(image_embed, image_embeds, dict_cat))
                else:  # If USER wants to do something else
                    raise InvalidRequest()
            else:  # If not
                # Getting the code to execute
                code_to_exec = self.openai_query(
                    prompt=intro
                    + "Write Python code that:\n"
                    + "1) imports any necessary libraries,\n"
                    + "2) creates an empty list named result,\n"
                    + "3) checks if table can be used to answer USER's request; if not, appends"
                    + "to result a Python string that explains to USER why not. If so,\n"
                    + "4) creates only pandas DataFrames/Series, Python f-strings, and/or "
                    + "Plotly Graph Objects as necessary that answer USER,\n"
                    + "5) appends to result the answer(s),\n"
                    + "6) and NEVER returns result.\n"
                    + "Some notes about the code:\n"
                    + "- Never write plain text, only Python code.\n"
                    + "- Never reference variables created in previous answers, "
                    + "since they do not persist after an answer is made.\n"
                    + "- Never create any functions or classes.\n"
                    + "- If asked to modify/lookup table, create a copy of table, write valid code to modify/lookup "
                    + "the copy instead while retaining as many rows and columns as possible, and return the copy.\n"
                    + "- Understand what happens when you call len() on a string or slice/call len() on a pandas object.\n"
                    + "- If USER asks for an image, call 'self.open_image()', which takes a string path "
                    + "to an image as input and returns the image as a Python numpy array, and append it to result. ",
                    temperature=0.3,
                    max_tokens=self.max_code_tokens,
                )
                print(code_to_exec + "\n\n\n\n\n")

                # Add the code to execute to the list of outputs
                outputs[0] = code_to_exec
                vars = {
                    "table": table,
                    "self": self,
                }
                answer = self.exec_code(vars, code_to_exec)

                # Check the answer
                if answer:
                    # Type check the answer
                    for output in answer:
                        if type(output) == pd.DataFrame:
                            outputs[1].append(output)
                        elif type(output) == str:
                            outputs[2].append(output)
                        elif type(output) == plotly.graph_objects.Figure:
                            outputs[3].append(output)
                        elif type(output) == np.ndarray:
                            outputs[4].append(self.img_to_str(output))
                        else:
                            raise ValueError()
                else:
                    raise ValueError()

            # Check if anything besides None in output
            if any(outputs):
                return outputs
            else:
                raise InvalidRequest()

        except InvalidRequest:  # Invalid question -> use pandas-profiling to generate a report
            message = self.invalid_request_error + self.report_message
            outputs[5] = self.get_report(table, message)
            return outputs

        except Exception:  # Something went wrong -> use pandas-profiling to generate a report
            print(traceback.format_exc())
            message = self.other_error + self.report_message
            outputs[5] = self.get_report(table, message)
            return outputs
