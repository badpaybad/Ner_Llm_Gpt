#See https://aka.ms/containerfastmode to understand how Visual Studio uses this Dockerfile to build your images for faster debugging.
#FROM ubuntu:focal
#FROM mcr.microsoft.com/dotnet/aspnet:6.0-focal
FROM nvidia/cuda:12.3.1-devel-ubuntu20.04

#RUN cat /etc/os-release
 RUN apt-get update -y && apt-get install -y nano curl
# RUN apt-get update -y && apt-get install -y nano \
#     apt-transport-https ca-certificates software-properties-common \
#     libfontconfig1 libfreetype6     
# #libfontconfig1 libfreetype6 # for skiasharp ( system drawing )

# #cause .net need pythonnet_netstandard_py38_linux
# ##RUN dotnet remove /src/MilionsFaceIdsApi.FaceidsPython/MilionsFaceIdsApi.FaceidsPython.csproj package pythonnet_netstandard_py38_win
# RUN add-apt-repository ppa:deadsnakes/ppa -y && apt update
# RUN apt install -y python3.8 libpython3.8 libpython3.8-stdlib python3-pip 
# RUN pip3 install --upgrade pip && pip3 install numpy

WORKDIR /app

# if manual download just uncomment this , and comment curl
COPY llava-v1.5-7b-q4-server.llamafile /app/llava-v1.5-7b-q4-server.llamafile

# RUN curl -LO https://huggingface.co/jartine/llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-q4-server.llamafile
RUN chmod 755 /app/llava-v1.5-7b-q4-server.llamafile

EXPOSE 8080
#https://simonwillison.net/2023/Nov/29/llamafile/
CMD [ "/app/llava-v1.5-7b-q4-server.llamafile", "--host","0.0.0.0","--port","8080" ,"--nobrowser"]
#--platform linux/amd64 
#sudo docker build --no-cache -f dockerfile -t docker.io/dunp/llamacpp .

#sudo docker run --gpus all  -d --restart always -p 8080:8080 --name llamacpp_8880 docker.io/dunp/llamacpp
#sudo docker run --gpus all  -d --restart always -p 8680:8080 --name llamacpp_8080_1 docker.io/dunp/llamacpp

#sudo docker build --no-cache -f dockerfile -t docker.io/dunp/llamacpp .
#docker run  -d --restart always -p 8080:8080 --name llamacpp_8880 docker.io/dunp/llamacpp

### install docker
#curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
# sudo add-apt-repository \
#    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
#    $(lsb_release -cs) stable"
# sudo apt-get update
# sudo apt-get install docker-ce docker-ce-cli containerd.io
#sudo usermod -aG docker $USER
## #
#nvidia-smi
#### nvidia docker suppot
# curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
# curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
# sudo apt-get update
# sudo apt-get install nvidia-container-runtime

# sudo systemctl stop docker
# sudo systemctl start docker

#docker container ls 
#docker rm -f llamacpp_8880

# sudo groupadd docker

# sudo usermod -aG docker $USER

# newgrp docker 
# sudo docker logs -f llamacpp_8880