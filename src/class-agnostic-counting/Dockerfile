FROM tensorflow/tensorflow:1.6.0-gpu-py3

RUN apt-get update
RUN apt-get install python3-tk

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

WORKDIR /app