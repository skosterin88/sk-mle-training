FROM ubuntu:latest
LABEL maintainer="Sergei Kosterin <skosterin88@gmail.com>"

RUN apt-get update 
RUN apt-get install -y sudo curl git zsh wget nano
RUN apt-get install -y python-is-python3 python3 pip python3-pip
RUN apt-get clean
RUN adduser --quiet --disabled-password --shell /bin/zsh --home /home/devuser --gecos User devuser
RUN echo "devuser:p@ssword1" | chpasswd 
RUN usermod -aG sudo devuser
    
# Copy source files
COPY requirements.txt config.yaml /home/devuser

WORKDIR /home/devuser

USER devuser
ENV TERM=xterm

RUN pip install -r requirements.txt --break-system-packages --no-cache-dir

CMD ["zsh"]
