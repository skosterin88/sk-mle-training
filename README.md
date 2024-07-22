# sk-mle-training
A repo used for ML Engineering course tasks.

**containerization**

A call to HuggingFace Inference API using the YOLOS-Small  object detection model ([https://api-inference.huggingface.co/models/hustvl/yolos-small]) on a test image with cats.
The Dockerfile contains commands for building a container named sk-hw-containerization. 
To build this container, use the following command:

```docker build -t sk-hw-containerization```

To run this container, use the following command:

```docker run sk-hw-containerization```
