#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
docker run -d -it --rm -p 8888:8888 -v $DIR:/home/jovyan/work  jupyter/datascience-notebook
