# captafied

<https://user-images.githubusercontent.com/40700820/214430538-da18d31c-2e7e-4511-a307-80f0903e61a4.mov>

## Contents

- [captafied](#captafied)
  - [Contents](#contents)
  - [Description](#description)
    - [Inference Pipeline](#inference-pipeline)
    - [Usage](#usage)
  - [Development](#development)
    - [Contributing](#contributing)
    - [Setup](#setup)
    - [Repository Structure](#repository-structure)
    - [Workflows](#workflows)
    - [Code Style](#code-style)
  - [Credit](#credit)

## Description

A website that helps users understand their spreadsheet data without the learning curve of data processing and visualization tools such as Excel or Python.

### Inference Pipeline

The user must submit a table and corresponding request regarding it. Optionally, there is an option to upload an image for similarity search, classification, etc. Then, we use [OpenAI's API](#credit) to generate Python code that returns one or more of the following:

- pandas DataFrames
- Python strings/f-strings
- Plotly graphs
- Images

that can be used to answer the user's request. If something fails in this process, we use [pandas-profiling](#credit) to generate a descriptive table profile that can be used to help the user understand their data.

### Usage

Some notes about submitting inputs to the pipeline:

- Only [long-form data](https://seaborn.pydata.org/tutorial/data_structure.html#long-form-vs-wide-form-data) is currently supported because we rely on [OpenAI's API](#credit) for many tasks, which doesn't actually see the data itself. Rather, it only has access to the variables associated with the data.
- Tables can only be submitted as .csv, .xls(x), .tsv, and .ods files.
- Images can only be submitted as .png, .jpeg and .jpg, .webp, and non-animated GIF (.gif).
- Only up to 150,000 rows and 30 columns of data can be submitted at one time.

Some examples of requests and questions that the pipeline can handle (these use the example table found in the repo and the website):

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

## Development

### Contributing

To contribute, check out the [guide](./CONTRIBUTING.md).

### Setup

1. Install conda if necessary:

   ```bash
   # Install conda: https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation
   # If on Windows, install chocolately: https://chocolatey.org/install. Then, run:
   # choco install make
   ```

2. Create the conda environment locally:

   ```bash
   cd captafied
   make conda-update
   conda activate captafied
   make pip-tools
   export PYTHONPATH=.
   echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
   ```

3. Install pre-commit:

   ```bash
   pre-commit install
   ```

4. Sign up for an OpenAI account and get an API key [here](https://beta.openai.com/account/api-keys).
5. Populate a `.env` file with your key and the backend URL in the format of `.env.template`, and reactivate the environment.
6. (Optional) Sign up for an AWS account [here](https://us-west-2.console.aws.amazon.com/ecr/create-repository?region=us-west-2) and set up your AWS credentials locally, referring to [this](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html#cli-configure-quickstart-config) as needed:

   ```bash
   aws configure
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
    ├── load_test   # the load testing code using Locust.
    ├── monitoring  # the model monitoring code
├── frontend        # the frontend code using Dash.
├── tasks           # the pipeline testing code.
```

### Workflows

- To start the app locally (uncomment code in `PredictorBackend.__init__` and set `use_url=False` to use the local model instead of the API):

  ```bash
  python frontend/app.py
  ```

- To login to AWS before deploying:

  ```bash
  . ./backend/deploy/aws_login.sh
  ```

- To deploy the backend to AWS Lambda:

  ```bash
  python backend/deploy/aws_lambda.py
  ```

### Code Style

- To lint the code:

  ```bash
  pre-commit run --all-files
  ```

## Credit

- OpenAI for their [API](https://openai.com/api/).
- YData for their [pandas-profiling](https://github.com/ydataai/pandas-profiling) package.
