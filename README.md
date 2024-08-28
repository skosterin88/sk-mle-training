# sk-mle-training
A repo used for ML Engineering course tasks.

**containerization**

A container that allows for running custom code on custom data inside the container.

To build the container, use the following command: 

```docker build --rm -f Dockerfile -t ubuntu:skosterin88 .
```

To run the container that allows to save the output file to the current working directory, use the following command: 

```docker run --rm -it -v `pwd`:/shared-volume ubuntu:skosterin88
```

It mounts the contents of the directory with your code to the shared-volume directory inside the container. 


The Python files inside the sk_hw_containerization directory contain the code necessary to call HuggingFace Inference API on any image you want; the test image is cats.jpg.
To run this code, do the following:

1) Grant the read-write access to everyone using the sk_hw_containerization folder:

```chmod -R a+rw ./
```

2) Run the container from inside the sk_hw_containerization folder:

```docker run --rm -it -v `pwd`:/shared-volume ubuntu:skosterin88
```

3) Navigate to the shared-volume folder inside the container:

```cd /shared-volume
```

4) Run the Python code: 

```python3 -m main <YOUR_IMAGE_FILE>
```

After that, the output will be saved into the output.json file in the folder on your host system. 



