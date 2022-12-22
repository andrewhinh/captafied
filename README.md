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
A full-stack ML-powered website that helps users understand their spreadsheet data without the learning curve of data processing and visualization tools such as Excel or Python. Regardless of whether your data includes numbers, text, or image links, answers are answered through automatically-generated sliced tables, text, plots, and HTML pages. 

## Inference Pipeline
The pipeline involves the following steps:
1. If the user wants to modify the table, they can specify how in natural language. We then use [OpenAI's API](#credit) to convert the command into a pandas query to modify the table accordingly.
2. If the user has a question about the table, they can ask it in natural language. As decided by [OpenAI's API](#credit):
    - If the question requires a numerical or text answer, we use [Google's Tapas](#credit) through the HF Inference API to answer the question.
    - If the question requires a graph, we use [OpenAI's API](#credit) to code up a reasonable graph to display, and [OpenAI's CLIP](#credit) to compute image and/or text embeddings as necessary and applicable.
    - If the question requires a HTML page, we use [pandas-profiling](#credit) to generate a descriptive table profile.
## Usage
Some examples of requests and questions that the pipeline can handle:
- Modification Request: 
    - Find all the repos that have more than 900 stars.
    - Add 10 stars to all the repos that have more than 900 stars.
- Simple Questions: 
    - How many stars does the transformers repo have?
    - Which repo has the most stars?
- Univariate Graph Question: 
    - What does the distribution of the repos' stars look like?
    - What does the distribution of the repos' summaries look like?
    - What does the distribution of the repos' icons look like?
- Multivariate Graph Question: 
    - What is the relationship between stars and forks?
    - How do stars, forks, and release year relate?
- Report Question:
    - What is the missing values situation for this table?
    - What is the duplicate rows situation for this table?

# Production
To setup the production server for the website in an AWS EC2 instance, we:
1. Setup the instance: install packages such as `pip`, pull the repo, and install the environment requirements:
2. Setup the Gradio app with an AWS Lambda backend:
```bash
python3 frontend/gradio/app.py --flagging --model_url=AWS_LAMBDA_URL
```
3. Serve the Gradio app over a permanent localtunnel link:
```bash
. ./frontend/localtunnel.sh
```
4. Implement continual development by updating the AWS Lambda backend when signaled by a pushed commit to the repo and checking if the pipeline's performance has improved:
```bash
. ./backend/deploy/cont_deploy.sh
```

# Development
## Setup
### Note
If the instructions aren't working for you, head to [this Google Colab](https://colab.research.google.com/drive/1Z34DLHJm1i1e1tnknICujfZC6IaToU3k?usp=sharing), make a copy of it, and run the cells there to get an environment set up.
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
3. Sign up for a HuggingFace account and get an access token [here](https://huggingface.co/settings/tokens).
4. Populate a `.env` file with your OpenAI API key and HuggingFace access token in the format of `.env.template`, and reactivate the environment.
5. Sign up for an AWS account [here](https://us-west-2.console.aws.amazon.com/ecr/create-repository?region=us-west-2) and set up your AWS credentials locally, referring to [this](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-config) as needed:
```bash
aws configure
```
6. Sign up for a Weights and Biases account [here](https://wandb.ai/signup) and download the CLIP ONNX file locally:
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
- Google for their [Table QA code](https://huggingface.co/google/tapas-base-finetuned-wtq).
- OpenAI for their [CLIP text and image encoder code](https://huggingface.co/openai/clip-vit-base-patch16) and [GPT-3 API](https://openai.com/api/).
- YData for their [pandas-profiling](https://github.com/ydataai/pandas-profiling) package.
