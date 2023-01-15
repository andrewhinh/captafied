# [captafied](https://captafied.loca.lt/)

<https://user-images.githubusercontent.com/40700820/212492942-4511107d-5e9c-415d-a253-8ecdefc3f3b8.mov>

## Contents

- [captafied](#captafied)
  - [Contents](#contents)
  - [Contributors](#contributors)
  - [Description](#description)
    - [Inference Pipeline](#inference-pipeline)
    - [Usage](#usage)
  - [Production](#production)
  - [Development](#development)
    - [Setup](#setup)
      - [Note](#note)
      - [Steps](#steps)
    - [Repository Structure](#repository-structure)
    - [Testing](#testing)
  - [Credit](#credit)

## Contributors

1. Andrew Hinh (ajhinh@gmail.com)
2. Laith Darras (laith.s.darras@gmail.com)
3. Calvin Hoang (calvinhoang21403@gmail.com)
4. Edison Zhang (edisonzhangsw@gmail.com)
5. Albert Ho (almtho2003@gmail.com)
6. Jair Martinez (martinez.jair1224@gmail.com, 1999jairmartinez@gmail.com)
7. Brian Huynh (brianhuynh1028@gmail.com)

## Description

A full-stack ML-powered website that helps users understand their spreadsheet data regardless of the format and without the learning curve of data processing and visualization tools such as Excel or Python.

### Inference Pipeline

Once the user submits a table and a request regarding it, we first determine the kind of request. Then,

- If a table modification is requested, we use [OpenAI's API](#credit) to generate Python code to return the whole modified table.
- If a table row-wise lookup is requested or reasoning question is asked, we use [OpenAI's API](#credit) to generate Python code to return the sliced table.
- If a table cell-wise lookup is requested or reasoning question is asked, we use [OpenAI's API](#credit) to generate Python code to return a conversational string that utilizes information in the table to answer the user.
- If a distribution or relationship is mentioned in the user's question, we use [OpenAI's API](#credit) to generate Python matplotlib code to execute, displaying a graph.
- If text or image embeddings or clusters are asked to be displayed, we use [OpenAI's CLIP](#credit) to compute image and/or text embeddings and Python matplotlib to display the graph accordingly.
- If the question doesn't belong to any of the above formats, we use [pandas-profiling](#credit) to generate a descriptive table profile.

### Usage

Some examples of requests and questions that the pipeline can handle, assuming the question references the input table:

- Table Modifications:
  - Add 10 stars to all the repos that have summaries longer than 10 words and icons with a height larger than 500 pixels.
  - Add a column named 'Stars_Forks' that averages the number of stars and forks for each row.
- Table row-wise lookups/questions:
  - Which rows have summaries longer than 10 words?
- Table cell-wise lookups/questions:
  - Does the Transformers repo have the most stars?
  - What is the shape of the Transformers repo's icon?
  - How many words long is the Transformers repo's description?
- Distribution/Relationship Questions:
  - What does the distribution of the stars look like?
- Text/Image Embedding/Cluster Questions:
  - How do the description embeddings change with release year?
  - How do the summary and icon embeddings change with number of stars?
- Report Questions:
  - How much memory does the dataset use?
  - How uniform are the columns?
  - What problems/challenges in the data need work to fix?

Some notes about submitting inputs to the pipeline:

- Only [long-form data](https://seaborn.pydata.org/tutorial/data_structure.html#long-form-vs-wide-form-data) is currently supported because we rely on [OpenAI's API](#credit) for many tasks, which doesn't actually see the data itself. Rather, it only has access to the variables associated with the data.
- Try to be clear what it is exactly that you're asking for; for example, to get the backend to properly understand you want text embeddings to be plotted, it may be necessary to specify as such in the request as seen in the examples above.
- By themselves, up to three continuous +/- categorical variables can be graphed at one time. When graphed with text and image embeddings, up to two continuous variables can be graphed. However, there is no limit on the number of text, image, and categorical variables that can be graphed.

## Production

To setup the production server for the website, we simply:

1. Setup the app with an AWS Lambda backend on our localhost:

```bash
python3 frontend/app.py --flagging --model_url=AWS_LAMBDA_URL
```

2. Serve the localhost app over a permanent localtunnel link:

```bash
. ./frontend/localtunnel.sh
```

## Development

### Setup

#### Note

- If the instructions aren't working for you, head to [this Google Colab](https://colab.research.google.com/drive/1Z34DLHJm1i1e1tnknICujfZC6IaToU3k?usp=sharing), make a copy of it, and run the cells there to get an environment set up.
- To contribute, reach out to Andrew @ ajhinh@gmail.com.

#### Steps

1. Set up the conda environment locally, referring to the instructions of the commented links as needed:

```bash
git clone https://github.com/andrewhinh/captafied.git
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

From the main directory, there are various ways to test the pipeline:

- To start the app locally:

```bash
python frontend/app.py --flagging
```

- To test the frontend by launching and pinging the frontend locally:

```bash
python -c "from frontend.tests.test_app import test_local_run; test_local_run()"
```

- To test various aspects of the model pipeline:

```bash
. ./tasks/REPLACE #replacing REPLACE with the corresponding shell script in the tasks/ folder
```

## Credit

- OpenAI for their [CLIP text and image encoder code](https://huggingface.co/openai/clip-vit-base-patch16) and [GPT-3 API](https://openai.com/api/).
- [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) for their dimensionality reduction algorithm.
- YData for their [pandas-profiling](https://github.com/ydataai/pandas-profiling) package.
