# Imports
import argparse
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import tabula as tb
from transformers import CLIPProcessor
import umap
import validators


# Setup
# matplotlib setup
matplotlib.use('agg') # Disable GUI for Gradio
plt.style.use('dark_background') # Dark background for plots

# Loading env variables
load_dotenv()

# OpenAI API setup
openai.organization = "org-SenjN6vfnkIealcfn6t9JOJj"
openai.api_key = os.getenv("OPENAI_API_KEY")

# Artifacts (models, etc.) path
artifact_path = Path(__file__).resolve().parent / "artifacts" / "inference"
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
        self.types = ["text", "image"]
        
    def openai_query(self, prompt, temperature, max_tokens, top_p, frequency_penalty=0.0, presence_penalty=0.0):
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

    def clip_encode(self, df, column_data):
        # Set up inputs for CLIP
        texts = []
        images = []

        if self.types[1] not in column_data.values(): # Only text data present
            texts = self.get_column_vals(df, column_data, self.types[0])
            images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(len(texts))] # Placeholder value
        elif self.types[0] not in column_data.values(): # Only image data present
            images = self.get_column_vals(df, column_data, self.types[1])
            texts = ["Placeholder value" for _ in range(len(images))]
        else: # Both image and text data present
            images = self.get_column_vals(df, column_data, self.types[1])
            texts = self.get_column_vals(df, column_data, self.types[0])
        
        # CLIP encoding
        inputs = self.clip_processor(text=texts, images=images, return_tensors="np", padding=True)
        clip_outputs = self.clip_session.run(
            output_names=["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"], input_feed=dict(inputs)
        )
        
        # Get embeds and prefix for graph title
        if self.types[1] not in column_data.values():
            list_embeds = [clip_outputs[2]]
            prefix = self.types[0][0].upper() + self.types[0][1:] + ' '
        elif self.types[0] not in column_data.values():
            list_embeds = [clip_outputs[3]]
            prefix = self.types[1][0].upper() + self.types[1][1:] + ' '
        else:
            list_embeds = [clip_outputs[2], clip_outputs[3]]
            prefix = self.types[0][0].upper() + self.types[0][1:] + ' and ' + self.types[1][0].upper() + self.types[1][1:] + ' '
        
        return list_embeds, prefix

    def open_image(self, image): 
        image_pil = Image.open(image)
        if image_pil.mode != "RGB": 
            image_pil = image_pil.convert(mode="RGB")
        return image_pil

    def get_column_vals(self, df, column_data, type):
        objects = []
        column_list = []
        for item in column_data.items():
            if item[1] == type: # Get column names for the specified data type
                column_list.append(item[0])
        for column in column_list: # Get values for the specified data type
            if type==self.types[1]:
                objects.extend([self.open_image(image) for image in df[column]])
            else:
                objects.extend(list(df[column]))
        return objects

    def get_embeds_graph(self, df, list_embeds, prefix):
        # Setting up matplotlib figure and legend
        _, ax = plt.subplots()
        plt.title(prefix + "Clusters")
        markers = itertools.cycle(("o", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", 
                                   "*", "h", "H", "+", "x", "X", "D", "d", 4, 5, 6, 7, 8, 9, 10, 11)) 
        colors = cm.rainbow(np.linspace(0, 1, len(list_embeds)))
        handles = []
        labels = []

        # UMAP and K-Means clustering
        offset = 0 # To label the clusters in a continuous manner
        for embeds, color in zip(list_embeds, colors):
            # Getting # of clusters and K-Means clustering
            range_n_clusters = list(range(2, len(df)))
            silhouette_scores = []
            for num_clusters in range_n_clusters:
                # initialise kmeans
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(embeds)
                cluster_labels = kmeans.labels_
                # silhouette score
                silhouette_scores.append(silhouette_score(embeds, cluster_labels))
            n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
            kmeans.fit(embeds)
            clusters = kmeans.labels_

            # Reducing dimensionality of embeddings with UMAP
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(embeds)
            x = embedding[:, 0]
            y = embedding[:, 1]

            # Plotting clusters
            for cluster in range(n_clusters):
                # Plotting points
                xs = np.array(x)[clusters == cluster]
                ys = np.array(y)[clusters == cluster]
                marker = next(markers)
                ax.scatter(xs, ys, color=color, marker=marker, alpha=1)
                
                # Adding cluster to legend
                artist = matplotlib.lines.Line2D([], [], color=color, lw=0, marker=marker)
                handles.append(artist)
                labels.append(str(cluster + offset))

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
        
        """
        # Bokeh
        from bokeh.colors import groups
        from bokeh.embed import json_item
        from bokeh.io import show
        from bokeh.plotting import figure
        p = figure(title = prefix + "Clusters")
        markers = itertools.cycle(('asterisk', 'circle', 'circle_cross', 'circle_dot', 'circle_x', 'circle_y', 'cross', 
                                   'dash', 'diamond', 'diamond_cross', 'diamond_dot', 'dot', 'hex', 'hex_dot', 
                                   'inverted_triangle', 'plus', 'square', 'square_cross', 'square_dot', 'square_pin', 
                                   'square_x', 'star', 'star_dot', 'triangle', 'triangle_dot', 'triangle_pin', 'x', 'y'))
        colors = []
        for name in groups.__all__:
            group = getattr(groups, name)
            colors.extend([x.to_hex() for x in group])
        handles = []
        labels = []

        # UMAP and K-Means clustering
        offset = 0 # To label the clusters in a continuous manner
        for embeds, color in zip(list_embeds, colors):
            # Getting # of clusters and K-Means clustering
            range_n_clusters = list(range(2, len(df)))
            silhouette_scores = []
            for num_clusters in range_n_clusters:
                # initialise kmeans
                kmeans = KMeans(n_clusters=num_clusters)
                kmeans.fit(embeds)
                cluster_labels = kmeans.labels_
                # silhouette score
                silhouette_scores.append(silhouette_score(embeds, cluster_labels))
            n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]
            kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
            kmeans.fit(embeds)
            clusters = kmeans.labels_

            # Reducing dimensionality of embeddings with UMAP
            reducer = umap.UMAP()
            embedding = reducer.fit_transform(embeds)
            x = embedding[:, 0]
            y = embedding[:, 1]

            # Plotting clusters
            for cluster in range(n_clusters):
                # Plotting points
                xs = np.array(x)[clusters == cluster]
                ys = np.array(y)[clusters == cluster]
                marker = next(markers)
                p.scatter(xs, ys, legend_label=str(cluster + offset), color=color, marker=marker, alpha=1)

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

        return json_item(p)
        """

    def predict(self, table: Union[str, Path, pd.DataFrame], request: Union[str, Path]) -> str:
        # Handling repeated uses of matplotlib
        plt.clf()

        # Type handling
        if not isinstance(table, pd.DataFrame):
            if "csv" in table.name:
                df = pd.read_csv(table.name) 
            elif "tsv" in table.name:
                df = pd.read_csv(table.name, sep='\t')
            elif "xlsx" in table.name:
                df = pd.read_excel(table.name) 
            elif "ods" in table.name:
                df = pd.read_excel(table.name, engine="odf")
            elif "pdf" in table.name:
                df = tb.read_pdf(table.name, pages='all')
            elif "html" in table.name:
                df = pd.read_html(table.name)                    
        else:
            df = table
        if isinstance(request, Path) | os.path.exists(request):
            with open(request, "r") as f:
                request_str = f.readline()
        else:
            request_str = request

        # Initialize result
        result = {}

        # Figure out what type of request is being made
        which_answer = self.openai_query(
            prompt="You are given the following text from a user: " + request_str + "\n" +
                    "You are also given a Python pandas DataFrame named df that has the following columns: " + 
                    ', '.join(list(df.columns)) + "\n" +
                    "The Python types of each column mentioned are listed in order: " +
                    ', '.join([str(type(df.loc[0, column])) for column in df.columns]) + "\n" +
                    "To reply to the user, you have three choices:\n" +
                    "1) Use OpenAI API's text completion endpoint to generate a Python pandas.query() query " +
                    "to return a modified copy/slice of df, text, or number/list.\n" +
                    "2) Use OpenAI API's text completion endpoint to generate code " +
                    "to return a Python matplotlib graph.\n" +
                    "3) Use Python's pandas-profiling package to return an HTML profile report of df.\n" +
                    "Assume that you know what each output is and must choose one as a reply to the user. " +
                    "Also assume that if the text from the user asks for a table modification/slice, " +
                    "text, or number/list, a query should be chosen as a reply. " +
                    "Lastly, assume that if the text from the user mentions a statistical relationship or " +
                    "distribution of data, a graph should be chosen as a reply. " +
                    "Write 1, 2, or 3 based on the corresponding format you choose: ",
            temperature=1,
            max_tokens=3,
            top_p=0.01,
        )
        which_answer = int(which_answer)

        # Check for image and text columns
        columns = self.openai_query(
            prompt="You are given the following question: " + 
                    request_str + "\n" +
                    "You are also given a Python pandas DataFrame named df that has the following columns: " + 
                    ', '.join(list(df.columns)) + "\n" +
                    "The Python types of each column mentioned are listed in order: " +
                    ', '.join([str(type(df.loc[0, column])) for column in df.columns]) + "\n" +
                    "List only the necessary columns that should be used " +
                    "to answer the question as a comma separated list: ",
            temperature=1,
            max_tokens=1000,
            top_p=0.001,
        )
        columns = columns.split(", ")
        column_data = {}
        for column in columns: 
            test = df.loc[0, column]
            if type(test) == str:
                if len(list(set(list(df[column])))) >= len(df) and len(test.split(" ")) > 3: # For text
                    column_data[column] = self.types[0]
                if validators.url(test): # For images
                    column_data[column] = self.types[1]
                elif path.exists(str(Path(__file__).resolve().parent / test)): # For local images
                    df[column] = df[column].apply(lambda x: str(Path(__file__).resolve().parent / x))
                    column_data[column] = self.types[1]

        # Generate a reply based on the type of request
        if which_answer == 1: # Use OpenAI's API to create a pd.query() statement 
            if column_data:
                image_columns = []
                for column in column_data:
                    if column_data[column] == self.types[1]:
                        image_columns.append(column)
                image_columns = ', '.join(image_columns)
                note = str("Note that the following columns have file paths/URLs to images: " + image_columns + "\n" +
                           "As such, call self.open_image(), which takes as input a file path/URL to an image and " +
                           "returns a PIL Image object, to open any images. To avoid type issues, " +
                           "convert the result of your pandas query, which may be a pd.DataFrame or pd.Series, " +
                           "into a list before calling self.open_image(). Then, perform any other necessary operations. ")
            else:
                note = ""
            
            mod_func = self.openai_query(
                prompt="You are given a Python pandas DataFrame named df that has the following columns: " + 
                        ', '.join(list(df.columns)) + "\n" +
                        "The Python types of each column mentioned are listed in order: " +
                        ', '.join([str(type(df.loc[0, column])) for column in df.columns]) + "\n" +
                        "Write a Python pandas .query() statement to fufill the following " +
                        "request/answer the following question: " + request_str + "\n" + note +
                        "Don't modify df in your statement, and only return the expression: ",
                temperature=0.3,
                max_tokens=60,
                top_p=1.0,
            )
            result = eval(mod_func)
            if type(result) != pd.DataFrame: # If the result is not a DataFrame
                if type(result) == pd.Series: # If the result is a Series
                    if len(result) == 1: # If the Series is a single element
                        result = str(result.iloc[0])
                    else: # If the Series is a list
                        result = ", ".join(str(element) for element in list(result))
                else: # If the result is a single element
                    result = str(result)

        elif which_answer == 2: 
            if column_data: # If there are image and/or text columns
                # Generating image and/or question embeddings and creating clustering graph(s)
                list_embeds, prefix = self.clip_encode(df, column_data)
                self.get_embeds_graph(df, list_embeds, prefix)
                result = plt
                # result = self.get_embeds_graph(df, list_embeds, prefix)                     
            else: # If there are no image or text columns
                graph_func = self.openai_query(
                    prompt="You are given the following question: " + 
                            request_str + "\n" +
                            "You are also given a Python pandas DataFrame named df that has the following columns: " + 
                            ', '.join(list(columns)) + "\n" +
                            "The Python types of each column mentioned are listed in order: " +
                            ', '.join([str(type(df.loc[0, column])) for column in columns]) + "\n" +
                            "Answer the question by writing Python Matplolib code to best intuitively graph the data in df. " +
                            "Do not include any imports and give the graph axis labels, a title, and a legend as necessary: ",
                    temperature=0.3,
                    max_tokens=150,
                    top_p=1.0,
                )
                exec(graph_func)
                result = plt

        elif which_answer == 3: # Use pandas-profiling to generate a report
            if len(df) > 1000:
                report = ProfileReport(df, title="Pandas Profiling Report", minimal=True)
            else:
                report = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
            
            # For Gradio
            report.config.html.inline = True
            report.config.html.minify_html = True
            report.config.html.use_local_assets = True
            report.config.html.navbar_show = False
            report.config.html.full_width = False
            result = report.to_html()

            """
            # For Dash
            report.to_file("/assets/report.html")
            result = "HTML" # Frontend will handle the logic here
            """
                
        return result


# Running model
def main():
    parser = argparse.ArgumentParser()

    # Inputs
    parser.add_argument("--table", type=str, required=True)
    parser.add_argument("--request", type=str)
    args = parser.parse_args()

    # Answering question
    pipeline = Pipeline()
    result = pipeline.predict(args.table, args.request) # Outputs the modified table, string, matplotlib graph, or HTML page

    print(result)


if __name__ == "__main__":
    main()
