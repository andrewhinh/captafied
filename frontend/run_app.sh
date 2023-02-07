#!/bin/bash
while ! docker run -it --rm captafied-frontend; do
    sleep 1
done