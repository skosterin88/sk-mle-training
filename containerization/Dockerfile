# syntax=docker/dockerfile:1

# Base image
FROM ubuntu:20.04

# Install dependencies
RUN apt-get -qq update && apt-get -y install python python3 pip python3-pip

WORKDIR /usr/bin/sk_hw_containerization

# Copy source files
COPY requirements.txt yolo_call.py main.py config.yaml cats.jpg ./
RUN pip install -r requirements.txt

#Launch app
CMD ["python3", "-m", "main"]

