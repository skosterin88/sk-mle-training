FROM ubuntu:latest
LABEL maintainer="Sergei Kosterin <skosterin88@gmail.com>"

RUN apt-get update && apt-get install -y sudo curl git zsh wget nano python-is-python3 python3 pip python3-pip && apt-get clean
RUN apt-get install -y python3-venv
RUN adduser --quiet --disabled-password --shell /bin/zsh --home /home/devuser --gecos User devuser
RUN echo "devuser:p@ssword1" | chpasswd 
RUN usermod -aG sudo devuser
RUN echo "devuser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
# RUN chsh -s /bin/zsh devuser

# Copy source files
COPY requirements.txt /home/requirements.txt

WORKDIR /home

USER devuser
ENV TERM=xterm

RUN python3 -m venv /home/devuser/.venv
ENV PATH="/home/devuser/.venv/bin:$PATH"

RUN echo "source /home/devuser/.venv/bin/activate" >> /home/devuser/.zshrc
RUN pip install --upgrade pip
# RUN pip install dvc
RUN sudo chown -R devuser /home
RUN sudo chmod -R u+rwX /home
RUN python3 -m pip install -r requirements.txt

CMD ["zsh"]