FROM tensorflow/tensorflow:1.6.0-gpu-py3

RUN apt-get update

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

WORKDIR /app
