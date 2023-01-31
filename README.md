# [captafied](https://captafied.loca.lt/)

<https://user-images.githubusercontent.com/40700820/214430538-da18d31c-2e7e-4511-a307-80f0903e61a4.mov>

## Contents

- [captafied](#captafied)
  - [Contents](#contents)
  - [Description](#description)
    - [Inference Pipeline](#inference-pipeline)
    - [Usage](#usage)
  - [Production](#production)
  - [Development](#development)
    - [Contributing](#contributing)
    - [Setup](#setup)
    - [Repository Structure](#repository-structure)
    - [Testing](#testing)
    - [Linting](#linting)
  - [Credit](#credit)

## Description

A website that helps users understand their spreadsheet data without the learning curve of data processing and visualization tools such as Excel or Python.

### Inference Pipeline

We provide the user with check-boxes to indicate if they want to use manually-implemented functions that perform the following tasks:

- Clustering (where text and/or images are grouped by similarity)
- Text search (where results are ranked by relevance to a query string)
- Image search (where results are ranked by relevance to a query string)
- Anomaly detection (where rows that have outliers are identified)

To be implemented:

- *Text classification (where text is classified by their most similar label)*
- *Image classification (where images are classified by their most similar label)*
  
Once the user submits a table, a request regarding it, and optionally checks one or more of the above boxes, we first check if the user wants to use a manually-implemented function.

If so, we use [OpenAI's CLIP](#credit) to compute image and/or text embeddings and [UMAP](#credit) to reduce the embeddings' dimensionality as necessary (it is assumed that for all cases besides anomaly detection, image and/or text is being submitted/referenced). Then, we call the corresponding manually-implemented function to perform the task.

Otherwise, we use [OpenAI's API](#credit) to generate Python code that returns one or more of the following:

- pandas DataFrames
- Python strings/f-strings
- Plotly graphs
- Images opened from URLs in the table

that can be used to answer the user's request. If something fails in this process, we use [pandas-profiling](#credit) to generate a descriptive table profile that can be used to help the user understand their data.

### Usage

Some notes about submitting inputs to the pipeline:

- Because past requests and answers are sent to [OpenAI's API](#credit), you can refer to past requests and answers to help formulate your current request, allowing for more complex requests.
- Because multiple outputs are supported, you can submit multiple requests at one time.
- Only [long-form data](https://seaborn.pydata.org/tutorial/data_structure.html#long-form-vs-wide-form-data) is currently supported because we rely on [OpenAI's API](#credit) for many tasks, which doesn't actually see the data itself. Rather, it only has access to the variables associated with the data.
- Only csv, xls(x), tsv, and ods files are currently supported.
- Only up to 150,000 rows and 30 columns of data can be submitted at one time.
- When submitting clustering requests, only clarifications for which columns and rows to use are accepted, since certain state variables cannot be stored due to memory limitations. In addition, up to two continuous variables can be graphed. However, there is no limit on the number of text, image, and categorical variables that can be graphed.
- When submitting text/image search/classification requests, preface the actual query within the request with a backslash ("\"). For example, if you wanted to find text in the 'Product_Description' column that is most similar to {query}, you could submit a request like "What is the most similar product description to \query?".
- After receiving the results from a manually-implemented function, you can reference them in every other request type (besides clustering for reasons described above).
- Try to explain any co-dependencies between columns that may exist. For example, assume there are two columns, 'Repository_Name' and 'Icon_URLs' and the 'Icon_URLs' column is a list of URLs that correspond to the icons of the repositories in the 'Repository_Name' column. In this case, you could explain this co-dependency in your request by saying something like "Show me the repo's icon." rather than "What does {repo} look like?".

Some examples of requests and questions that the pipeline can handle (with respect to the example table found in the repo and website):

- Add 10 stars to all the repos that have summaries longer than 10 words.
  - Of the repos you just added stars to, which ones have the most stars?
- Which rows have summaries longer than 10 words?
  - Of the rows you just selected, which ones were released in 2020?
- Does the Transformers repo have the most stars?
  - What about the least?
- What does the distribution of the stars look like?
  - Center the title.
- What does the Transformers icon look like?
  - Make it half as tall.
- How much memory does the dataset use?
  - What's this number in MB?
- How do the description clusters change with release year?
  - Plot this graph vs. the number of stars.
- Which rows have anomalies in the summary column?
  - What about every column?

## Production

To setup the production server for the website, we simply:

1. Setup the app with an AWS Lambda backend on our localhost:

```bash
python3 frontend/app.py --model_url=AWS_LAMBDA_URL
```

2. Serve the localhost app over a permanent localtunnel link:

```bash
. ./frontend/lt.sh
```

## Development

### Contributing

To contribute, check out the [guide](./CONTRIBUTING.md).

### Setup

1. Set up the conda environment locally, referring to the instructions of the commented links as needed:

```bash
cd captafied
# Install conda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
    # If on Windows, install chocolately: https://chocolatey.org/install. Then, run:
    # choco install make
make conda-update 
conda activate captafied
make pip-tools
export PYTHONPATH=.
echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
# If you're using a newer NVIDIA RTX GPU: 
    # pip3 uninstall torch torchvision torchaudio -y
    # Download the PyTorch version that is compatible with your machine: https://pytorch.org/get-started/locally/
```

2. Sign up for an OpenAI account and get an API key [here](https://beta.openai.com/account/api-keys).
3. Populate a `.env` file with your OpenAI API key in the format of `.env.template`, and reactivate the environment.
4. Sign up for an AWS account [here](https://us-west-2.console.aws.amazon.com/ecr/create-repository?region=us-west-2) and set up your AWS credentials locally, referring to [this](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-config) as needed:

```bash
aws configure
```

5. Sign up for a Weights and Biases account [here](https://wandb.ai/signup) and download the CLIP ONNX file locally:

```bash
wandb login
python ./backend/inference/artifacts/stage_model.py --fetch
```

If the instructions aren't working for you, head to [this Google Colab](https://colab.research.google.com/drive/1Z34DLHJm1i1e1tnknICujfZC6IaToU3k?usp=sharing), make a copy of it, and run the cells there to get an environment set up.

### Repository Structure

The repo is separated into main folders that each describe a part of the ML-project lifecycle, some of which contain interactive notebooks, and supporting files and folders that store configurations and workflow scripts:

```bash
.
├── backend   
    ├── deploy      # the AWS Lambda backend setup and continuous deployment code.
        ├── api_serverless  # the backend handler code using AWS Lambda.
    ├── inference   # the inference code.
        ├── artifacts   # the model (W&B-synced) storage folder.
    ├── load_test   # the load testing code using Locust.
    ├── monitoring  # the model monitoring code
├── frontend        # the frontend code using Dash.
├── tasks           # the pipeline testing code.
```

### Testing

- To start the app locally:

```bash
python frontend/app.py
```

### Linting

- To lint the code (after staging your changes):

```bash
make lint
```

## Credit

- OpenAI for their [CLIP text and image encoder code](https://huggingface.co/openai/clip-vit-base-patch16) and [GPT-3 API](https://openai.com/api/).
- [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) for their dimensionality reduction algorithm.
- YData for their [pandas-profiling](https://github.com/ydataai/pandas-profiling) package.
