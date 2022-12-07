# captafied

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
A full-stack ML-powered website that helps users understand their spreadsheet data whether it contains images, text, or numbers.

## Inference Pipeline
The pipeline involves the following steps:
1. If the user wants to modify the table, they can specify how in natural language. We then use [OpenAI's API](#credit) to convert the command into an SQL query to modify the table accordingly.
2. If the user has a question about the table, they can ask it in natural language. As decided by [OpenAI's API](#credit):
    - If the question requires a numerical or text answer, we use [Google's Tapas](#credit) through the HF Inference API to answer the question.
    - If the question requires a graph, we use [OpenAI's API](#credit) to pick a reasonable graph to display, and [OpenAI's CLIP](#credit) to compute image and/or text embeddings as necessary and applicable.
## Usage
Some examples of filter conditions and questions that the pipeline can handle:
- Modify Request: 
    - "Show me all the users whose name is John and age is greater than 20."
    - "Add 13 to every user who has more than 4 keys."
- Numerical Questions: 
    - "How many users were born in 1990?"
    - "What month is the weather the highest?"
- Univariate Graph Question: 
    - "What does the distribution of the number of orders look like?"
    - "What categories does the pasta come in?"
- Multivariate Graph Question: 
    - "What is the relationship between the number of users and the number of orders?"
    - "How does month, year, and number of customers relate?"

# Production
To setup the production server for the website in an AWS EC2 instance, we:
1. Setup the instance: install packages such as `pip`, pull the repo, and install the environment requirements:
2. Setup the Gradio app with an AWS Lambda backend:
```bash
python3 frontend/app.py --flagging --model_url=AWS_LAMBDA_URL
```
3. Serve the Gradio app over a permanent localtunnel link:
```bash
. ./frontend/localtunnel.sh
```
4. Implement continual development by updating the AWS Lambda backend when signaled by a pushed commit to the repo and checking if the pipeline performance has improved:
```bash
. ./backend/deploy/cont_deploy.sh
```
5. Implement continual training by running the training pipeline every ? weeks and checking if the pipeline performance has improved:
```bash
TBD
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
3. Sign up for OpenAI's API [here](https://openai.com/api/), populate a `.env` file with your OpenAI API key in the format of `.env.template`, and reactivate (just activate again) the environment.
4. Sign up for an AWS account [here](https://us-west-2.console.aws.amazon.com/ecr/create-repository?region=us-west-2) and setup your AWS credentials locally, referring to [this](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-config) as needed:
```bash
aws configure
```
5. Sign up for a Weights and Biases account [here](https://wandb.ai/signup) and download the models and context examples locally:
```bash
wandb login
python ./training/stage_model.py --fetch --from_project captafied
```
## Repository Structure
The repo is separated into main folders that each describe a part of the ML-project lifecycle, some of which contain interactive notebooks, and supporting files and folders that store configurations and workflow scripts:
```bash
.
├── api_serverless  # the backend handler code using AWS Lambda.
├── backend   
    ├── artifacts   # the model W&B-synced storage folder.
    ├── deploy      # the AWS Lambda backend setup and continuous deployment code.
    ├── inference   # the inference code.
    ├── load_test   # the load testing code using Locust.
    ├── monitoring  # the model monitoring code using Gradio's flagging feature.
├── frontend        # the frontend code using Gradio (for now).
├── tasks           # the pipeline testing code.
```
## Testing
From the main directory, there are various ways to test the pipeline:
- To start the Gradio app locally:
```bash
python frontend/app.py --flagging
```
- To test the Gradio frontend by launching and pinging the frontend locally:
```bash
python -c "from frontend.tests.test_app import test_local_run; test_local_run()"
```
- To test various aspects of the model pipeline:
```bash
. ./tasks/REPLACE #replacing REPLACE with the corresponding shell script in the tasks/ folder
```

# Credit
- Google for their [Table QA code](https://huggingface.co/google/tapas-base-finetuned-wtq).
- OpenAI for their [CLIP text and image encoder code](https://huggingface.co/openai/clip-vit-base-patch16) and [GPT-3 API](https://openai.com/api/).
