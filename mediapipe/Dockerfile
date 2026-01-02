# Base image Ubuntu 24.04
FROM ubuntu:24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

COPY . /app
COPY ./config.prod.py /app/config.py

RUN apt update && apt install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN python3 -m pip install --upgrade pip setuptools wheel

RUN apt-get update && apt-get install -y libxcb-xinerama0 libxcb-xinerama0-dev qt5-default libgtk-3-dev python3-dev libyaml-dev \ 
    libgtk2.0-dev pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0

RUN apt install -y python3.10 python3.10-venv python3.10-dev python3.10-distutils

RUN deactivate
RUN rm -rf venv
RUN python3.10 -m venv venv
RUN source venv/bin/activate
RUN pip install --upgrade pip setuptools wheel
RUN pip install tensorflow==2.13.1 mediapipe==0.10.9 mediapipe-model-maker==0.2.1.4

# deactivate
# rm -rf venv
# python3.10 -m venv venv
# source venv/bin/activate
# pip install --upgrade pip setuptools wheel
# pip install tensorflow==2.13.1
# pip install mediapipe==0.10.9 mediapipe-model-maker==0.2.1.4

# RUN python3 train.py 
# RUN python3 inference.py
EXPOSE 80
# Sử dụng dấu && để nối lệnh
CMD python3 train.py && python3 inference.py

# docker build -f "Dockerfile" -t dockerregistry.airobotics.vn/mldlai-test-img-clasify:dev .
# docker push dockerregistry.airobotics.vn/mldlai-test-img-clasify:dev
# sudo docker pull dockerregistry.airobotics.vn/mldlai-test-img-clasify:dev


# sudo docker rm -f mldlai-test-img-clasify-dev
# sudo docker run -d --restart=always  -p 9088:80 --name mldlai-test-img-clasify-dev dockerregistry.airobotics.vn/mldlai-test-img-clasify:dev

