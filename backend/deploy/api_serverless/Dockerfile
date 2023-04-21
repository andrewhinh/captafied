# Starting from an official AWS image
# Keep any dependencies and versions in this file aligned with the environment.yml and Makefile
FROM public.ecr.aws/lambda/python:3.8

# Install Python dependencies
COPY requirements/prod.txt ./requirements.txt
RUN pip install --upgrade pip==21.1.3
RUN pip install -r requirements.txt

# Copy only the relevant directories and files
#   note that we use a .dockerignore file to avoid copying logs etc.
#   UMAP cache files are stored in /tmp/numba_cache
#   reports are stored in /tmp/assets
ENV NUMBA_CACHE_DIR=/tmp/numba_cache
RUN mkdir -p /tmp/assets

COPY .env .env
COPY backend/inference/ ./backend/inference
COPY backend/deploy/api_serverless/api.py ./api.py
COPY utils/util.py ./utils/util.py

CMD ["api.handler"]
