# Imports
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import requests
from typing import Any, Dict, List, Tuple, Union

from dotenv import load_dotenv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from onnxruntime import InferenceSession
import openai
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from transformers import CLIPProcessor
import validators


# Variables
# Artifact path
artifact_path = Path(__file__).resolve().parent / "artifacts" / "inference"
onnx_path = artifact_path / "onnx"

# CLIP Encoders config
clip_processor = artifact_path / "clip-vit-base-patch16"
clip_onnx = onnx_path / "clip.onnx"

# Loading env variables
load_dotenv()

# OpenAI Engine
engine = "text-davinci-003"


# Main Class
class Pipeline:
    """
    Main inference class
    """

    def __init__(self):
        # CLIP Setup
        self.clip_session = InferenceSession(str(clip_onnx))
        self.clip_processor = CLIPProcessor.from_pretrained(clip_processor)

        # OpenAI API Setup
        openai.api_key = os.getenv("OPENAI_API_KEY")

        # HF API Setup
        self.TAPAS_API_URL = "https://api-inference.huggingface.co/models/google/tapas-base-finetuned-wtq"
        self.headers = {"Authorization": "Bearer " + os.getenv("HF_API_KEY")}

    def tapas_query(self, payload):
        response = requests.post(self.TAPAS_API_URL, headers=self.headers, json=payload)
        return response.json()

    def predict(self, table: Union[str, Path, pd.DataFrame], request: Union[str, Path]) -> str:
        # Type handling
        if not isinstance(table, pd.DataFrame): # Need to add more logic to handle different types of files (.csv, .xlsx, .json, .txt, .html, .pdf, etc.)
            df = pd.read_csv(table.name) 
        else:
            df = table
        if isinstance(request, Path) | os.path.exists(request):
            with open(request, "r") as f:
                request_str = f.readline()
        else:
            request_str = request

        request_type = openai.Completion.create(
            model=engine,
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
        if request_type == 'True':
            mod_func = openai.Completion.create(
                model=engine,
                prompt="You are given a Python pandas DataFrame named 'df' that has the following columns: " + 
                        ', '.join(list(df.columns)) + "\n" +
                        "Write a Python pandas function named 'f' to " + request_str + ". " + 
                        "Make sure to not return the entire function definition; " + 
                        "only return everything after and not including the 'return' statement: ",
                temperature=0.3,
                max_tokens=60,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )["choices"][0]["text"].strip()
            result = eval(mod_func)
            if type(result) != pd.DataFrame:
                result = result.to_frame()
        
        # Question Handling
        elif request_type == 'False': # Need to add more logic to check for image/text columns + speed of clustering + handle different types of plots
            result_type = openai.Completion.create(
                model=engine,
                prompt="You are given the following question: " + 
                        request_str + "\n" +
                        "You are also given a Python pandas DataFrame named 'df' that has the following columns: " + 
                        ', '.join(list(df.columns)) + "\n" +
                        "The Python types of each column mentioned are listed in order:" +
                        ', '.join([str(type(column)) for column in df.columns]) + "\n" +
                        "Write 'True' if it is possible to answer the question with a number, statement, or list, or " + 
                        "'False' otherwise, meaning a graph is required to answer the question: ",
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
            
            elif result_type == "False":
                columns = openai.Completion.create(
                    model=engine,
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
                for column in columns:
                    test = df.loc[0, column]
                    if type(test) == str:
                        if validators.url(test):
                            column_data[column] = "image"
                        elif len(test) > 3:
                            column_data[column] = "text"

                if column_data: # If there are image or text columns
                    images = []
                    texts = []
                    if "image" not in column_data.values(): # Only text data present
                        images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(len(df))]
                        texts = list(df[column])
                    elif "text" not in column_data.values(): # Only image data present
                        images = list(df[column])
                        texts = ["Placeholder value" for _ in range(len(df))]
                    
                    # Generating image and/or question embeddings
                    inputs = self.clip_processor(text=texts, images=images, return_tensors="np", padding=True)
                    clip_outputs = self.clip_session.run(
                        output_names=["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"], input_feed=dict(inputs)
                    )

                    if "image" not in column_data.values(): # Only text data present
                        embeds = clip_outputs[2]
                    elif "text" not in column_data.values(): # Only image data present
                        embeds = clip_outputs[3]
                    
                    # Creating clustering graph
                    # Calculating optimal number of clusters
                    range_n_clusters = list(range(1, 10))
                    silhouette_scores = []
                    for num_clusters in range_n_clusters:
                        # initialise kmeans
                        kmeans = KMeans(n_clusters=num_clusters)
                        kmeans.fit(embeds)
                        cluster_labels = kmeans.labels_
                        # silhouette score
                        silhouette_scores.append(silhouette_score(embeds, cluster_labels))
                    n_clusters = silhouette_scores.index(max(silhouette_scores))

                    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
                    kmeans.fit(embeds)
                    labels = kmeans.labels_

                    tsne = TSNE(
                        n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200
                    )
                    vis_dims2 = tsne.fit_transform(embeds)

                    cats = []
                    for category in range(n_clusters):
                        vis_dims = np.array(vis_dims2)[labels == category]
                        cats.append(vis_dims.assign(dataset=category))
                    cats = pd.concat(cats)

                    result = sns.scatterplot(data=cats, style='dataset')
                else: # No image or text data present
                    plot_func = openai.Completion.create(
                        model=engine,
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

    if type(result) == pd.DataFrame:
        print(result.head())
    elif type(result) == str:
        print(result)
    elif type(result) == sns.Figure:
        result.show()
    else:
        print("Couldn't show result")


if __name__ == "__main__":
    main()
