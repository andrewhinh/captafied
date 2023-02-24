#!/bin/bash
AWS_REGION=$(aws configure get region)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account | sed 's/"//g')
while ! docker run -p 11700:11700 -it --rm "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/captafied-frontend:latest"; do
    sleep 1
done