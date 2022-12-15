# Imports
import argparse
import os
from pathlib import Path
import requests
from typing import Union

from dotenv import load_dotenv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from onnxruntime import InferenceSession
import openai
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import tabula as tb
from transformers import CLIPProcessor
import validators


# Setup
# Disabling matplotlib GUI for Gradio
matplotlib.use('agg')

# Loading env variables
load_dotenv()

# OpenAI API Setup
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

        # HF API Setup
        self.TAPAS_API_URL = "https://api-inference.huggingface.co/models/google/tapas-large-finetuned-wtq"
        self.headers = {"Authorization": "Bearer " + os.getenv("HF_API_KEY")}
        
    def tapas_query(self, payload):
        response = requests.post(self.TAPAS_API_URL, headers=self.headers, json=payload)
        return response.json()

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

        # Figure out what type of request is being made
        request_type = openai.Completion.create(
            model=self.engine,
            prompt="You are given the following sentence: " + 
                    request_str + "\n" +
                    "Write True if the sentence ends with a period or exclamation point or is a statement, and " + 
                    "False if the sentence ends with a question mark or is a question: ",
            temperature=0,
            max_tokens=3,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )["choices"][0]["text"].strip()
        result = {}

        # Table Modification Handling
        if request_type == 'True': # Create a pd.query() function with OpenAI's API
            mod_func = openai.Completion.create(
                model=self.engine,
                prompt="You are given a Python pandas DataFrame named 'df' that has the following columns: " + 
                        ', '.join(list(df.columns)) + "\n" +
                        "Write a Python pandas '.query()' statement to " + request_str + ". " + 
                        "Don't modify `df` in your statement: ",
                temperature=0.3,
                max_tokens=60,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )["choices"][0]["text"].strip()
            result = eval(mod_func)
            if type(result) != pd.DataFrame: # In case pd.Series is returned
                result = result.to_frame()
        
        # Question Handling
        elif request_type == 'False': # First figure out what columns are needed to answer the question
            columns = openai.Completion.create(
                    model=self.engine,
                    prompt="You are given the following question: " + 
                            request_str + "\n" +
                            "You are also given a Python pandas DataFrame named 'df' that has the following columns: " + 
                            ', '.join(list(df.columns)) + "\n" +
                            "List the columns that should be used to answer the question as a comma separated list: ",
                    temperature=0.3,
                    max_tokens=60,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )["choices"][0]["text"].strip()
            columns = columns.split(", ")
            column_data = {}
            for column in columns: # Check for image and text columns
                test = df.loc[0, column]
                if type(test) == str:
                    if validators.url(test):
                        column_data[column] = "image"
                    if len(list(set(list(df[column])))) >= len(df) and len(test.split(" ")) > 3:
                        column_data[column] = "text"
                    
            if column_data: # If there are image or text columns
                # Preparing data for CLIP
                images = []
                texts = []
                image_present = False
                text_present = False
                if "image" not in column_data.values(): # Only text data present
                    text_present = True
                    images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(len(df))]
                    texts = list(df[column])
                elif "text" not in column_data.values(): # Only image data present
                    image_present = True
                    images = list(df[column])
                    texts = ["Placeholder value" for _ in range(len(df))]
                
                # Generating image and/or question embeddings
                inputs = self.clip_processor(text=texts, images=images, return_tensors="np", padding=True)
                clip_outputs = self.clip_session.run(
                    output_names=["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"], input_feed=dict(inputs)
                )
                if text_present:
                    embeds = clip_outputs[2]
                elif image_present:
                    embeds = clip_outputs[3]
                
                # Creating clustering graph
                # Calculating optimal number of clusters
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

                # Generating labels and visualizing clusters
                kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
                kmeans.fit(embeds)
                labels = kmeans.labels_

                tsne = TSNE(
                    n_components=2, perplexity=15, init="random", learning_rate=200
                )
                vis_dims2 = tsne.fit_transform(embeds)
                x = [x for x, y in vis_dims2]
                y = [y for x, y in vis_dims2]

                for category in range(n_clusters):
                    xs = np.array(x)[labels == category]
                    ys = np.array(y)[labels == category]
                    plt.scatter(xs, ys, alpha=0.3)

                    avg_x = xs.mean()
                    avg_y = ys.mean()

                    plt.scatter(avg_x, avg_y, marker="x", s=100)
                
                plt.title("Clusters identified and visualized in language 2d using t-SNE")
                result = plt
            
            else: # Figure out whether to use TAPAS or not
                result_type = openai.Completion.create( 
                    model=self.engine,
                    prompt="You are given the following question: " + 
                            request_str + "\n" +
                            "You are also given a Python pandas DataFrame named 'df' that has the following columns: " + 
                            ', '.join(list(df.columns)) + "\n" +
                            "The Python types of each column mentioned are listed in order:" +
                            ', '.join([str(type(column)) for column in df.columns]) + "\n" +
                            "Write 'False' if the question mentions statistical relationships, " + 
                            "distributions of data, or categorical data, and 'True' otherwise: ",
                    temperature=0,
                    max_tokens=3,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )["choices"][0]["text"].strip()

                if result_type == "True":
                    df_to_json = df.to_dict(orient="list")
                    for col in df_to_json:
                        df_to_json[col] = [str(x) for x in df_to_json[col]]
                    while 'cells' not in list(result.keys()):
                        result = self.tapas_query({
                            "inputs": {
                                "query": request_str,
                                "table": df_to_json,
                            },
                        })
                    result = result['cells']
                    result = ", ".join(result)
                
                elif result_type == "False": # Create a plot using Matplotlib and OpenAI
                    plot_func = openai.Completion.create(
                        model=self.engine,
                        prompt="You are given the following question: " + 
                                request_str + "\n" +
                                "You are also given a Python pandas DataFrame named 'df' that has the following columns: " + 
                                ', '.join(list(df.columns)) + "\n" +
                                "The Python types of each column mentioned are listed in order:" +
                                ', '.join([str(type(column)) for column in df.columns]) + "\n" +
                                "Answer the question by writing Python Matplolib code to best plot the data" + ". " +
                                "Make sure to separate newlines with semicolons, and not include any imports: ",
                        temperature=0.3,
                        max_tokens=500,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0
                    )["choices"][0]["text"].strip()
                    exec(plot_func)
                    result = plt

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
    result = pipeline.predict(args.table, args.request) # Outputs the modified table, string, or seaborn plot

    print(result)


if __name__ == "__main__":
    main()
