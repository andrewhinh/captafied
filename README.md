# [captafied](https://captafied.loca.lt/)
![demo](./demo.png)

# Contents
- [Description](#description)
    - [Inference Pipeline](#inference-pipeline)
    - [Usage](#usage)
- [Production](#production)
- [Development](#development)
    - [Setup](#setup)
    - [Repository Structure](#repository-structure)
    - [Testing](#testing)
- [Credit](#credit)

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
. ./frontend/gradio/localtunnel.sh
```
4. Implement continual development by updating the AWS Lambda backend when signaled by a pushed commit to the repo and checking if the pipeline performance has improved:
```bash
. ./backend/deploy/cont_deploy.sh
```
5. Implement continual training by running the training pipeline every ? weeks and checking if the pipeline performance has improved:
```bash
. ./backend/deploy/cont_train.sh
```

# Development
## Setup
1. Follow the steps listed [here](https://github.com/full-stack-deep-learning/fsdl-text-recognizer-2022-labs/tree/main/setup#local), replacing the corresponding commands with:
```bash
git clone https://github.com/andrewhinh/captafied.git
cd captafied
conda activate captafied
```
2. If you're using a newer NVIDIA RTX GPU, uninstall PyTorch and visit [here](https://pytorch.org/get-started/locally/) to download the PyTorch version that is compatible with your machine:
```bash
pip3 uninstall torch torchvision torchaudio -y
```
3. Sign up for an OpenAI's API account [here](https://openai.com/api/).
4. Sign up for a HuggingFace account [here](https://huggingface.co/).
5. Populate a `.env` file with your OpenAI and HuggingFace API keys in the format of `.env.template`, and reactivate (just activate again) the environment.
6. Sign up for an AWS account [here](https://us-west-2.console.aws.amazon.com/ecr/create-repository?region=us-west-2) and setup your AWS credentials locally, referring to [this](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-config) as needed:
```bash
aws configure
```
7. Sign up for a Weights and Biases account [here](https://wandb.ai/signup) and download the models and context examples locally:
```bash
wandb login
python ./backend/inference/artifacts/stage_model.py --fetch --from_project captafied
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
