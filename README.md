# [captafied](https://captafied.loca.lt/)
![demo](./demo.png)

# Contents
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

# Contributors
1. Andrew Hinh (ajhinh@gmail.com)
2. Akhil Devarasetty (akhild2004@gmail.com)
3. Laith Darras (laith.s.darras@gmail.com)
4. Calvin Hoang (calvinhoang21403@gmail.com)
5. Edison Zhang (edisonzhangsw@gmail.com)
6. Albert Ho (almtho2003@gmail.com)
7. Jair Martinez (martinez.jair1224@gmail.com, 1999jairmartinez@gmail.com)
8. Brian Huynh (brianhuynh1028@gmail.com)

# Description
A full-stack ML-powered website that helps users understand their spreadsheet data without the learning curve of data processing and visualization tools such as Excel or Python. Regardless of whether your data includes numbers, text, or image links, answers are answered through automatically-generated sliced tables, text, plots, and/or HTML pages. 

## Inference Pipeline
Once the user submits a table and a text regarding it, we first determine the format of the answer we need to generate. Then,
- If the text requires a table, number, or text, we use [OpenAI's API](#credit) to create a pandas .query() statement and parse the result accordingly. 
    - We do this instead of using [OpenAI's API](#credit) to directly generate Python code to answer the question because we want to provide the user with as much information as possible unless directed to do otherwise. For example, if the user asks for the repo with the most stars, we want to provide them with the repo name and the number of stars, not just the repo name. However, if the user asks for the number of stars for the repo with the most stars, we want to provide them with just the number of stars.
- If the text requires a graph, we use [OpenAI's API](#credit) to generate matplotlib code to execute, displaying a grpah. If applicable, we also use [OpenAI's CLIP](#credit) to compute image and/or text embeddings.
- If the text requires an HTML page, we use [pandas-profiling](#credit) to generate a descriptive table profile.
## Usage
Some examples of requests and questions that the pipeline can handle:
- Table Requests/Questions: 
    - Find all the repos that have more than 900 stars.
    - Add 10 stars to all the repos that have more than 900 stars.
    - Which repo has the most forks?
- Numerical/Text Questions: 
    - How many forks does the CLIP repo have?
    - How many words long is the Flax repo's description?
    - Does the Transformers repo have the most stars compared to the other repos?
    - What are the number of stars for repos with more than 900 stars?
- Univariate Graph Questions: 
    - What does the distribution of the repos' stars look like?
    - What does the distribution of the repos' summaries look like?
    - What does the distribution of the repos' icons look like?
- Multivariate Graph Questions: 
    - What is the relationship between stars and forks?
    - How do stars, forks, and release year relate?
    - How do the distributions of the repos' summaries and descriptions compare?
    - How do the distributions of the repos' summaries, descriptions, and icons compare?
- Report Questions:
    - What is the missing values situation for this table?
    - What is the duplicate rows situation for this table?

# Production
To setup the production server for the website, we simply:
1. Setup the Gradio app with an AWS Lambda backend on our localhost:
```bash
python3 frontend/gradio/app.py --flagging --model_url=AWS_LAMBDA_URL
```
2. Serve the localhost app over a permanent localtunnel link:
```bash
. ./frontend/localtunnel.sh
```

# Development
## Setup
### Note
- If the instructions aren't working for you, head to [this Google Colab](https://colab.research.google.com/drive/1Z34DLHJm1i1e1tnknICujfZC6IaToU3k?usp=sharing), make a copy of it, and run the cells there to get an environment set up.
- To contribute, reach out to Andrew @ ajhinh@gmail.com.
### Steps
1. Set up the conda environment locally, referring to the instructions of the commented links as needed:
```bash
git clone https://github.com/andrewhinh/captafied.git
cd captafied
# Install conda: https://conda.io/projects/conda/en/latest/user-guide/install/linux.html
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
## Repository Structure
The repo is separated into main folders that each describe a part of the ML-project lifecycle, some of which contain interactive notebooks, and supporting files and folders that store configurations and workflow scripts:
```bash
.
├── backend   
    ├── deploy      # the AWS Lambda backend setup and continuous deployment code.
        ├── api_serverless  # the backend handler code using AWS Lambda.
    ├── inference   # the inference code.
        ├── artifacts   # the model (W&B-synced) storage folder.
    ├── load_test   # the load testing code using Locust.
    ├── monitoring  # the model monitoring code using Gradio's flagging feature.
├── frontend        
    ├── gradio      # Gradio frontend.
    ├── dash        # Dash frontend.
├── tasks           # the pipeline testing code.
```
## Testing
From the main directory, there are various ways to test the pipeline:
- To start the Gradio app locally:
```bash
python frontend/gradio/app.py --flagging
```
- To test the Gradio frontend by launching and pinging the frontend locally:
```bash
python -c "from frontend.gradio.tests.test_app import test_local_run; test_local_run()"
```
- To test various aspects of the model pipeline:
```bash
. ./tasks/REPLACE #replacing REPLACE with the corresponding shell script in the tasks/ folder
```

# Credit
- OpenAI for their [CLIP text and image encoder code](https://huggingface.co/openai/clip-vit-base-patch16) and [GPT-3 API](https://openai.com/api/).
- YData for their [pandas-profiling](https://github.com/ydataai/pandas-profiling) package.
