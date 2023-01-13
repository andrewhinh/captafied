# Imports
import argparse
import base64
import io
import itertools
import os
from os import path
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from onnxruntime import InferenceSession
import openai
import pandas as pd
from pandas_profiling import ProfileReport
from PIL import Image
import requests
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tabula as tb
from transformers import CLIPProcessor
import umap
import validators


# Setup
# matplotlib setup
# matplotlib.use('agg') # Disable GUI for Gradio                                        # Uncomment for Gradio
plt.style.use("dark_background")  # Dark background for plots

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


# Main class
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

        # Matplotlib setup
        self.types = ["text string", "image path/URL", "categorical", "continuous"]

    def is_string_series(self, s: pd.Series):
        if isinstance(s.dtype, pd.StringDtype):
            # The series was explicitly created as a string series (Pandas>=1.0.0)
            return True
        elif s.dtype == "object":
            # Object series, check each value
            return all((v is None) or isinstance(v, str) for v in s)
        else:
            return False

    def openai_query(
        self, prompt, temperature=1.0, max_tokens=16, top_p=1.0, frequency_penalty=0.0, presence_penalty=0.0
    ):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        return response["choices"][0]["text"].strip()

    def open_image(self, image):
        image_pil = Image.open(image)
        if image_pil.mode != "RGB":
            image_pil = image_pil.convert("RGB")
        return np.array(image_pil)

    def exec_code(self, df, code):
        global_vars, local_vars = {"self": self, "df": df}, {}
        exec(code, global_vars, local_vars)
        try:
            result = local_vars["result"]
            return result
        except:
            pass

        # In case of graph

    def get_column_vals(self, df, column, images_present=False):
        if images_present:
            return [self.open_image(image) for image in df[column]]
        else:
            return list(df[column])

    def clip_encode(self, df, text_columns, image_columns):
        # Set up inputs for CLIP
        texts = []
        images = []
        if text_columns:
            for column in text_columns:
                texts.append(self.get_column_vals(df, column))
        if image_columns:
            for column in image_columns:
                images.append(self.get_column_vals(df, column, images_present=True))

        if not texts:
            clip_images = list(itertools.chain.from_iterable(images))
            clip_texts = ["Placeholder value" for _ in range(len(clip_images))]
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

        # Get columns and embeds
        dict_embeds = {}
        if text_columns:
            for column in text_columns:
                dict_embeds[column] = text_embeds[: len(df[column])]
                text_embeds = text_embeds[len(df[column]) :]
        if image_columns:
            for column in image_columns:
                dict_embeds[column] = image_embeds[: len(df[column])]
                image_embeds = image_embeds[len(df[column]) :]

        return dict_embeds

    def get_embeds_graph(self, df, column_data, columns):
        # Separate columns by type
        text_columns = []
        image_columns = []
        cat_columns = []
        cont_columns = []
        for column in columns:
            if column_data[column] in self.types[0]:
                text_columns.append(column)
            elif column_data[column] in self.types[1]:
                image_columns.append(column)
            elif column_data[column] == self.types[2]:
                cat_columns.append(column)
            elif column_data[column] == self.types[3]:
                cont_columns.append(column)
            else:
                pass

        # CLIP embeddings of text and/or images and getting values of any categorical/continuous columns
        dict_embeds = self.clip_encode(df, text_columns, image_columns)
        dict_cat = {}
        dict_cont = {}
        if cat_columns:
            for column in cat_columns:
                dict_cat[column] = self.get_column_vals(df, column)
        if cont_columns:
            for column in cont_columns:
                dict_cont[column] = self.get_column_vals(df, column)
        other_vars_present = dict_cat or dict_cont
        two_other_vars_present = len(dict_cat) == 2 or len(dict_cont) == 2 or (dict_cat and dict_cont)

        # Setting up matplotlib figure
        if other_vars_present:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        else:
            fig, ax = plt.subplots()
        variables = list(dict_embeds.keys()) + list(dict_cat.keys()) + list(dict_cont.keys())
        variables = [variable + " Clusters" if variable in dict_embeds.keys() else variable for variable in variables]
        variables = " vs. ".join(variables)
        plt.title(variables)

        # Setting up legend
        colors = cm.rainbow(np.linspace(0, 1, len(dict_embeds)))
        markers = itertools.cycle(
            (
                "o",
                "v",
                "^",
                "<",
                ">",
                "1",
                "2",
                "3",
                "4",
                "8",
                "s",
                "p",
                "P",
                "*",
                "h",
                "H",
                "+",
                "x",
                "X",
                "D",
                "d",
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
            )
        )
        cluster_handles = []
        cluster_labels = []
        column_handles = []
        column_labels = []

        # UMAP and K-Means clustering
        offset = 0  # To label the clusters in a continuous manner
        for column, embeds, color in zip(list(dict_embeds.keys()), list(dict_embeds.values()), colors):
            # Getting # of clusters and K-Means clustering
            range_n_clusters = list(range(2, len(df)))
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

            # Reducing dimensionality of embeddings with UMAP
            n_neighbors = 15
            n_components = 2
            if embeds.shape[0] < 15:  # UMAP's default n_neighbors=15, reduce if # of data points is less than 15
                n_neighbors = embeds.shape[0] - 1
            if two_other_vars_present:  # Reduce UMAP n_components to accomodate for 2 cat/cont variables
                n_components = 1
            reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
            embedding = reducer.fit_transform(embeds)
            if n_components == 1:
                x = embedding
                y = None
            else:
                x = embedding[:, 0]
                y = embedding[:, 1]

            # Plotting clusters
            for cluster in range(n_clusters):
                # Plotting points
                marker = next(markers)
                xs = np.array(x)[clusters == cluster]  # Text/image data

                # Extending added variables to match the number of data points
                if dict_cat:
                    for item in dict_cat.items():
                        dict_cat[item[0]] = item[1] * int(len(clusters) / len(item[1]))
                if dict_cont:
                    for item in dict_cont.items():
                        dict_cont[item[0]] = item[1] * int(len(clusters) / len(item[1]))

                if two_other_vars_present:  # 2 categorical/continuous variables
                    if len(dict_cat) == 1:  # 1 categorical and 1 continuous variable
                        ys = np.array(list(dict_cont.values())[0])[clusters == cluster]
                        zs = np.array(list(dict_cat.values())[0])[clusters == cluster]
                        ax.set_ylabel(list(dict_cont.keys())[0])
                        ax.set_zlabel(list(dict_cat.keys())[0])
                        for zs_cat in np.unique(zs):
                            xs_cat = xs[zs == zs_cat]
                            ys_cat = ys[zs == zs_cat]
                            ax.scatter(xs_cat, ys_cat, zs_cat, zdir="z", color=color, marker=marker, alpha=1)
                    elif len(dict_cat) == 2:  # 2 categorical variables
                        ys = np.array(list(dict_cat.values())[0])[clusters == cluster]
                        zs = np.array(list(dict_cat.values())[1])[clusters == cluster]
                        ax.set_ylabel(list(dict_cat.keys())[0])
                        ax.set_zlabel(list(dict_cat.keys())[1])
                        for ys_cat in np.unique(ys):
                            for zs_cat in np.unique(zs):
                                filter = np.logical_and([ys == ys_cat], [zs == zs_cat])
                                xs_cat = xs[filter[0]]
                                ax.scatter(xs_cat, ys_cat, zs_cat, zdir="z", color=color, marker=marker, alpha=1)
                    elif len(dict_cont) == 2:  # 2 continuous variables
                        ys = np.array(list(dict_cont.values())[0])[clusters == cluster]
                        zs = np.array(list(dict_cont.values())[1])[clusters == cluster]
                        ax.set_ylabel(list(dict_cont.keys())[0])
                        ax.set_zlabel(list(dict_cont.keys())[1])
                        ax.scatter(xs, ys, zs, color=color, marker=marker, alpha=1)
                    else:
                        pass
                else:  # If < 2 continuous/categorical variables are present
                    if other_vars_present:  # If 1 continuous/categorical variable is present
                        ys = np.array(y)[clusters == cluster]
                        if dict_cat:  # 1 categorical variable
                            zs = np.array(list(dict_cat.values())[0])[clusters == cluster]
                            ax.set_zlabel(list(dict_cat.keys())[0])
                            for zs_cat in np.unique(zs):
                                xs_cat = xs[zs == zs_cat]
                                ys_cat = ys[zs == zs_cat]
                                ax.scatter(xs_cat, ys_cat, zs_cat, zdir="z", color=color, marker=marker, alpha=1)
                        elif dict_cont:  # 1 continuous variable
                            zs = np.array(list(dict_cont.values())[0])[clusters == cluster]
                            ax.set_zlabel(list(dict_cont.keys())[0])
                            ax.scatter(xs, ys, zs, color=color, marker=marker, alpha=1)
                        else:
                            pass
                    else:  # If no continuous/categorical variables are present
                        ys = np.array(y)[clusters == cluster]
                        ax.scatter(xs, ys, color=color, marker=marker, alpha=1)

                # Adding cluster to legend
                artist = matplotlib.lines.Line2D([], [], color=color, lw=0, marker=marker)
                cluster_handles.append(artist)
                cluster_labels.append(str(cluster + offset))

            # To label the clusters in a continuous manner
            offset += n_clusters

            # Adding column to legend
            artist = matplotlib.lines.Line2D([], [], color=color, lw=0, marker="o")
            column_handles.append(artist)
            column_labels.append(column)

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
            for color, type in zip(colors, self.types):
                artist = matplotlib.lines.Line2D([], [], color=color, lw=0, marker="o")
                handles.append(artist)
                labels.append(type[0].upper() + type[1:])
            ax.legend(handles, labels, loc="lower left", title="Data Types")
        """

        # Legends
        legend = ax.legend(cluster_handles, cluster_labels, loc="upper right", title="Clusters")
        ax.add_artist(legend)
        ax.legend(column_handles, column_labels, loc="lower left", title="Columns")

        # return plt                                                                    # Uncomment for Gradio

    def get_report(self, df, message):
        if len(df) > 1000:
            report = ProfileReport(df, title="Pandas Profiling Report", minimal=True)
        else:
            report = ProfileReport(df, title="Pandas Profiling Report", explorative=True)

        """
        # For Gradio
        report.config.html.inline = True
        report.config.html.minify_html = True
        report.config.html.use_local_assets = True
        report.config.html.navbar_show = False
        report.config.html.full_width = False
        return error + report.to_html()
        """

        # For Dash
        report_path = Path("assets") / "report.html"
        full_report_path = parent_path / ".." / ".." / "frontend" / "dash" / report_path
        report.to_file(full_report_path)
        return [message, "/" + str(report_path)]

    def predict(self, table: Union[str, Path, pd.DataFrame], request: Union[str, Path]) -> str:
        # Handling repeated uses of matplotlib
        plt.clf()

        # Empty return values for both successes and failures
        empty_pred_table, empty_pred_text, empty_pred_graph, empty_pred_report, empty_err = None, None, None, None, None

        # Type handling
        if not isinstance(table, pd.DataFrame):
            try:
                table.name = table.name.replace("/edit#gid=", "/export?format=csv&gid=")  # In case this is a url
                if "csv" in table.name:
                    df = pd.read_csv(table.name)
                elif "tsv" in table.name:
                    df = pd.read_csv(table.name, sep="\t")
                elif "xlsx" in table.name or "xls" in table.name:
                    df = pd.read_excel(table.name)
                elif "ods" in table.name:
                    df = pd.read_excel(table.name, engine="odf")
                elif "pdf" in table.name:
                    df = tb.read_pdf(table.name, pages="all")
                elif "html" in table.name:
                    df = pd.read_html(table.name)
                else:
                    raise ValueError(
                        "File type not supported, please submit a public Google Sheets URL or a csv/tsv/xls(x)/"
                        + "ods/pdf/html file."
                    )
            except ValueError as error:
                return empty_pred_table, empty_pred_text, empty_pred_graph, empty_pred_report, str(error)
            except:
                return (
                    empty_pred_table,
                    empty_pred_text,
                    empty_pred_graph,
                    empty_pred_report,
                    str("Sorry, we don't know what went wrong."),
                )
        else:
            df = table
        if isinstance(request, Path) | os.path.exists(request):
            with open(request, "r") as f:
                request_str = f.readline()
        else:
            if isinstance(request, str):
                request_str = request
            else:
                return (
                    empty_pred_table,
                    empty_pred_text,
                    empty_pred_graph,
                    empty_pred_report,
                    str("Sorry, we don't know what went wrong."),
                )

        # Getting data types of columns mentioned in the question
        column_data = {}
        for col_idx in range(len(df.columns)):  # For each column
            column = df.columns[col_idx]  # Get the column name
            test = df[column]  # Get the column values
            if len(set(test)) >= len(df):  # For continuous data
                if self.is_string_series(test):  # For strings
                    for row_idx in range(len(test)):  # For each row
                        value = test[row_idx]  # Get the value
                        potential_path = str(parent_path / value)  # Get the potential path
                        if validators.url(value):  # For images
                            df.at[row_idx, column] = requests.get(value, stream=True).raw
                            if not column in column_data:
                                column_data[column] = self.types[1]
                        elif path.exists(potential_path):  # For local images
                            df.at[row_idx, column] = potential_path
                            if not column in column_data:
                                column_data[column] = self.types[1]
                        else:  # For text
                            column_data[column] = self.types[0]
                            break  # Break out of the loop at the first text value
                else:  # For continuous data
                    column_data[column] = self.types[3]
            else:  # For categorical data
                column_data[column] = self.types[2]

        # Get the column names, types, and example values
        df_info = [[item[0], item[1], df.loc[0, item[0]]] for item in column_data.items()]
        df_info = [", ".join([str(item) for item in column]) for column in df_info]
        df_info = "; ".join(df_info)

        # Get the user's request type
        which_answer = self.openai_query(
            prompt="You are given a Python pandas DataFrame named df. The following is a list of its columns, "
            + "their data types, and an example value from each one: "
            + df_info
            + "\n"
            + "A user asks the following from you regarding df: "
            + request_str
            + "\n"
            + "The five kinds of requests users can make are the following:\n"
            + "- Table modifications, which encompass every form of modification of the table,\n"
            + "- Table row-wise lookups/reasoning questions, which involve no modification of the table and "
            + "ask for entire rows that satisfy some condition,\n"
            + "- Table cell-wise lookups/reasoning questions, which involve no modification of the table and "
            + "ask for cells that satisfy some condition,\n"
            + "- Distribution/relationship questions without text or image clusters/embeddings, which ask for "
            + "patterns within the table.\n"
            + "- Questions involving text and/or image embeddings/clusters, which ask for patterns within the "
            + "table.\n"
            + "Return '1', '2', '3', '4', or '5' based on which kind of request you think the user is making. "
            + "Return '0' if you can't tell or the request doesn't belong to any of the above kinds: ",
            temperature=0,
            max_tokens=3,
        )

        try:  # For valid questions
            # Converting to int
            which_answer = int(which_answer)

            # Define notes for OpenAI prompt
            note = str(
                "Some notes about writing the code:\n"
                + "1. Don't call 'print()' or 'return'.\n"
                + "2. Whenever you call 'len()' or slice a pandas DataFrame, understand what it will return.\n"
                + "3. Avoid creating functions or classes unless necessary, in which case they must be called "
                + "within the code.\n"
                + "4. Import any necessary libraries, but don't import anything else.\n"
                + "5. As necessary, call 'self.open_image()', which takes a string path to an image as an argument "
                + "and returns the image as a Python numpy array, either directly on an string path or as a mapped "
                + "function over a list or pandas Series.\n"
                + "6. If the user is asking a question that cannot be answered with the information found in df, "
                + "return 'raise ValueError()'. "
            )

            # Table modifications
            if which_answer == 1:
                code_to_exec = self.openai_query(
                    prompt="You are given a Python pandas DataFrame named df. The following is a list of its "
                    + "columns, their data types, and an example value from each one: "
                    + df_info
                    + "\n"
                    + "A user asks the following from you regarding df: "
                    + request_str
                    + "\n"
                    + "Write Python code that creates a copy of df to modify while retaining the whole table and "
                    + "assigns the whole table to result. "
                    + note,
                    temperature=0.1,
                    max_tokens=250,
                )
                return self.exec_code(df, code_to_exec), empty_pred_text, empty_pred_graph, empty_pred_report, empty_err

            # Table row-wise lookups/reasoning questions
            elif which_answer == 2:
                code_to_exec = self.openai_query(
                    prompt="You are given a Python pandas DataFrame named df. The following is a list of its "
                    + "columns, their data types, and an example value from each one: "
                    + df_info
                    + "\n"
                    + "A user asks the following from you regarding df: "
                    + request_str
                    + "\n"
                    + "Write Python code that creates a copy of df to slice by row and assigns the sliced table to "
                    + "result. "
                    + note,
                    temperature=0.1,
                    max_tokens=250,
                )
                return self.exec_code(df, code_to_exec), empty_pred_text, empty_pred_graph, empty_pred_report, empty_err

            # Table cell-wise lookups/reasoning questions
            elif which_answer == 3:
                code_to_exec = self.openai_query(
                    prompt="You are given a Python pandas DataFrame named df. The following is a list of its "
                    + "columns, their data types, and an example value from each one: "
                    + df_info
                    + "\n"
                    + "A user asks the following from you regarding df: "
                    + request_str
                    + "\n"
                    + "Write Python code that first finds relevant information from df, then generates a string "
                    + "that uses the information to answer the user's request and assigns it to result. Make sure "
                    + "to use f-strings correctly and accurately and conversationally answer the user's request in "
                    + "the string. For example, if the user asks 'Does the Transformers repo have the most stars?' "
                    + "and the Transformers repo has 17000 stars, the most stars of any repo, don't write 17000, "
                    + "but instead write somethinglike '\"Yes, the Transformers repo has the most stars with 17000 "
                    + "stars.\"'. "
                    + note,
                    temperature=0.1,
                    max_tokens=250,
                )
                return (
                    empty_pred_table,
                    self.exec_code(df, code_to_exec).strip('"'),
                    empty_pred_graph,
                    empty_pred_report,
                    empty_err,
                )

            # Distribution/relationship questions without text or image clusters
            elif which_answer == 4:
                code_to_exec = self.openai_query(
                    prompt="You are given a Python pandas DataFrame named df. The following is a list of its "
                    + "columns, their data types, and an example value from each one: "
                    + df_info
                    + "\n"
                    + "A user asks the following from you regarding df: "
                    + request_str
                    + "\n"
                    + "Write Python code that draws an appropriate graph with matplotlib, doesn't include any "
                    + "imports, labels the graph's title, axes, and legend as necessary. "
                    + note,
                    temperature=0.3,
                    max_tokens=250,
                )
                """                                                                 # Add this back into prompt for Gradio
                , and ends with the line " +
                        "'result = plt'. 
                
                """
                # self.exec_code(df, code_to_exec)                                  # Replace return value with this for Gradio
                # """                                                                # Remove # for Gradio
                buf = io.BytesIO()
                self.exec_code(df, code_to_exec)
                plt.savefig(buf, format="png")
                plt.close()
                data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
                buf.close()
                value = "data:image/png;base64,{}".format(data)
                # """                                                                # Remove # for Gradio
                return empty_pred_table, empty_pred_text, value, empty_pred_report, empty_err

            # Questions involving text and/or image embeddings/clusters
            elif which_answer == 5:
                columns = self.openai_query(
                    prompt="You are given a Python pandas DataFrame named df. The following is a list of its "
                    + "columns, their data types, and an example value from each one: "
                    + df_info
                    + "\n"
                    + "A user asks the following from you regarding df: "
                    + request_str
                    + "\n"
                    + "List the columns mentioned in the user request that are necessary to generate a graph to "
                    + "answer the user as a comma-separated list. Some notes:\n"
                    + "1. If a mentioned column has the words 'embeddings' after it, it is most likely a text "
                    + "or image column and needs to be included.\n"
                    + "2. If you think a column is necessary but it is phrased in a way that suggests it posseses "
                    + "another column, it should be ignored. For example, if the user asks 'What do the repo's "
                    + "description embeddings look like?' with the current table df, the column 'Repos' should be "
                    + "ignored because it posseses the column 'Description', and only the column 'Description' "
                    + "should be used.\n"
                    + "3. If the user is asking to graph more than two categorical and/or continuous columns, "
                    + "return 'None'. ",
                    temperature=0.1,
                    max_tokens=250,
                )
                columns = columns.split(", ")
                for column in columns:
                    if column not in df.columns:
                        raise ValueError("Invalid request being made.")
                # self.get_embeds_graph(df, column_data, columns)                   # Replace return value with this for Gradio
                # """                                                                # Remove # for Gradio
                buf = io.BytesIO()
                self.get_embeds_graph(df, column_data, columns)
                plt.savefig(buf, format="png")
                plt.close()
                data = base64.b64encode(buf.getbuffer()).decode("utf8")  # encode to html elements
                buf.close()
                value = "data:image/png;base64,{}".format(data)
                # """                                                                # Remove # for Gradio
                return empty_pred_table, empty_pred_text, value, empty_pred_report, empty_err

            else:
                raise ValueError()

        except ValueError:  # Invalid question -> use pandas-profiling to generate a report
            message = str(
                "I don't know how to answer that question. Here's a report on the table generated by YData's "
                + "pandas-profiling library that might help you. "
            )
            return empty_pred_table, empty_pred_text, empty_pred_graph, self.get_report(df, message), empty_err

        except:  # Something went wrong -> use pandas-profiling to generate a report
            message = str(
                "Something went wrong. Here's a report on the table generated by YData's pandas-profiling "
                + "library that might help you. "
            )
            return empty_pred_table, empty_pred_text, empty_pred_graph, self.get_report(df, message), empty_err


# Running model
def main():
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument("--table", type=str, required=True)
    parser.add_argument("--request", type=str)
    args = parser.parse_args()

    # Answering question
    pipeline = Pipeline()
    result = pipeline.predict(
        args.table, args.request
    )  # Outputs the modified table, string, matplotlib graph, or HTML page

    print(result)


if __name__ == "__main__":
    main()
