# Use python official images. Slime or alpine is not recommended, 
# the official image already inlcudes git
# Source: https://pythonspeed.com/articles/base-image-python-docker-images/
FROM  nvcr.io/nvidia/jax:23.08-py3 
# Setup env
ENV LANG C.UTF-8
# ENV LC_ALL C.UTF-8
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONFAULTHANDLER 1

# COPY files
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
